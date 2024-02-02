# ----------------------------------------------------------------------
#
# File: AbstractDataTypes.py
#
# Last edited: 25.04.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
from collections import namedtuple
from typing import Dict, Iterable, Union

import numpy as np

_POINTERSYMBOL = "*"

IntegerType = namedtuple('IntegerType', ['width', 'signed'])
FutureType = namedtuple('FutureType', ['valueType', 'stateReferenceType', 'resolveTemplate', 'dispatchTemplate'])


class ImmediateType(type):

    def __new__(cls, name, bases, namespace):
        assert "typeName" in namespace.keys(), "Missing typeName in immediate type constructor!"
        assert "typeWidth" in namespace.keys(), "Missing typeWidth in immediate type constructor!"

        retCls = super().__new__(cls, name, bases, namespace)

        return retCls

    @classmethod
    def __prepare__(cls, name, bases):
        return dict()

    def __instancecheck__(cls, other):
        if hasattr(other, "typeName") and hasattr(other, "typeWidth"):
            return cls.typeName == other.typeName and cls.typeWidth == other.typeWidth
        return False


class IntegerImmediateType(ImmediateType):

    def __new__(cls, name, bases, namespace):
        assert "typeMax" in namespace.keys(), "Missing typeMax in immediate type constructor!"
        assert "typeMin" in namespace.keys(), "Missing typeMin in immediate type constructor!"

        retCls = super().__new__(cls, name, bases, namespace)
        return retCls

    def __instancecheck__(cls, other):
        if not super().__instancecheck__(other):
            return False

        if hasattr(other, "typeMax") and hasattr(other, "typeMin"):
            return cls.typeMin == other.typeMin and cls.typeMax == other.typeMax
        return False


class StructType(type):

    class StructDict(dict):

        def __init__(self):
            super().__init__()
            self._members = {}

        def __setitem__(self, key, value):
            if isinstance(value, (ImmediateType, StructType, PointerType)):
                self._members[key] = value
            else:
                super().__setitem__(key, value)

    def __new__(cls, name, bases, namespace):
        retCls = super().__new__(cls, name, bases, namespace)

        assert "typeName" in namespace.keys(), "Missing typeName in struct type constructor!"
        assert "typeWidth" in namespace.keys(), "Missing typeWidth in struct type constructor!"
        assert "structTypeDict" in namespace.keys(), "Missing structTypeDict in struct type constructor!"

        return retCls

    @classmethod
    def __prepare__(cls, name, bases):
        return StructType.StructDict()

    def __instancecheck__(cls, other):

        if not (hasattr(other, "typeName") and hasattr(other, "typeWidth") and hasattr(other, "structTypeDict")):
            return False

        if not (cls.typeName == other.typeName and cls.typeWidth == other.typeWidth):
            return False

        return cls.structTypeDict == other.structTypeDict


class PointerType(type):

    def __new__(cls, name, bases, namespace):
        retCls = super().__new__(cls, name, bases, namespace)

        assert "typeName" in namespace.keys(), "Missing typeName in pointer type constructor!"
        assert "typeWidth" in namespace.keys(), "Missing typeWidth in pointer type constructor!"
        assert "referencedType" in namespace.keys(), "Missing referencedType in pointer type constructor!"

        return retCls

    def __instancecheck__(cls, other):
        if hasattr(other, "typeName") and hasattr(other, "typeWidth"):
            return cls.typeName == other.typeName and cls.typeWidth == other.typeWidth


class _DataTypeCollection(type):

    class TypeDict(dict):

        def __init__(self):
            super().__init__()
            self._members = {}

        def __setitem__(self, key, value):
            if isinstance(value, IntegerType):
                if value.signed:
                    self._members[key] = IntegerImmediateType(
                        key, (IntegerImmediateClass,), {
                            "typeName": key,
                            "typeWidth": value.width,
                            "typeMax": 2**(value.width - 1) - 1,
                            "typeMin": -2**(value.width - 1)
                        })
                else:
                    self._members[key] = IntegerImmediateType(key, (IntegerImmediateClass,), {
                        "typeName": key,
                        "typeWidth": value.width,
                        "typeMax": 2**(value.width) - 1,
                        "typeMin": 0
                    })
            elif isinstance(value, dict):
                width = 0
                for _type in value.values():
                    width += _type.typeWidth
                self._members[key] = StructType(key, (StructClass,), {
                    "typeName": key,
                    "typeWidth": width,
                    "structTypeDict": value
                })
            else:
                super().__setitem__(key, value)

        def update(self, other):
            for key, value in other.items():
                if isinstance(value, (ImmediateType, StructType, PointerType)):
                    self._members[key] = value
                else:
                    self.__setitem__(key, value)

    def __new__(cls, name, bases, namespace):
        retCls = super().__new__(cls, name, bases, namespace)
        # SCHEREMO: The members field is used to store all generated types in the collection
        retCls._members = namespace._members
        return retCls

    def __add__(cls, otherCls):
        meta = type(cls)
        members = type(cls).TypeDict()
        members.update(cls._members)
        members.update(otherCls._members)
        newCls = meta("ComposedTypes", (cls, otherCls), members)
        return newCls

    def __getattr__(cls, key):
        return cls._members[key]

    def __iter__(cls):
        return (cls._members[key] for key in cls._members.keys())

    @classmethod
    def __prepare__(cls, name, bases):
        return _DataTypeCollection.TypeDict()


class DataTypeCollection(metaclass = _DataTypeCollection):
    pass


class ImmediateClass(metaclass = ImmediateType):

    __slots__ = ["value"]
    typeName: str = None
    typeWidth: int = None

    @classmethod
    def checkValue(cls, value, ctxt = None):
        return True

    @classmethod
    def _checkValue(cls, value: Union[int, float, ImmediateClass], ctxt = None):

        # Value promotion
        if issubclass(type(value), ImmediateClass):
            _value = value.value
        # Value assignment
        else:
            _value = value

        return cls.checkValue(_value, ctxt)

    def __init__(self, value: Union[int, float, ImmediateClass, ImmediateType], ctxt = None):

        assert self._checkValue(value), f"Cannot assign {value} to a {self.typeName}"
        self.value = value

    def __eq__(self, other):
        if not (isinstance(self, type(other)) and hasattr(other, "value")):
            return False

        return self.value == other.value

    def __repr__(self):
        return f"{str(self.value)}"


class IntegerImmediateClass(ImmediateClass, metaclass = IntegerImmediateType):

    __slots__ = ["value"]
    typeName: str = None
    typeWidth: int = None
    typeMax: Union[int, float] = None
    typeMin: Union[int, float] = None

    @classmethod
    def checkValue(cls, value: Union[int, np.array], ctxt = None):
        if not isinstance(value, Iterable):
            _max, _min = (value, value)
        else:
            _max = value.max()
            _min = value.min()

        if _max > cls.typeMax:
            return False
        if _min < cls.typeMin:
            return False
        return True


class PointerClass(metaclass = PointerType):

    __slots__ = ["referenceName", "_mangledReferenceName"]
    typeName: str = None
    typeWidth: int = None
    referencedType: Union[ImmediateType, StructType, PointerType] = None

    @classmethod
    def checkValue(cls, value: str, ctxt: NetworkContext) -> bool:
        if ctxt is None:
            return False

        if value is None or value == "NULL":
            print(f"WARNING: Setting pointer value to NULL - Referenced data is invalid!")
            return True

        reference = ctxt.lookup(value)
        _type = reference._type

        if _type is None:
            if not hasattr(reference, value):
                return True
            return cls.referencedType._checkValue(reference.value)

        if not isinstance(cls, Pointer(HelperTypes.void)) and not isinstance(_type.referencedType, cls.referencedType):
            return False

        return True

    @classmethod
    def _checkValue(cls, _value: Union[str, PointerClass], ctxt: NetworkContext) -> bool:
        if issubclass(type(_value), PointerClass):
            value = _value.referenceName
        else:
            value = _value
        return cls.checkValue(value, ctxt)

    def __init__(self, _value: str, ctxt: NetworkContext):

        if _value is not None and not self._checkValue(_value, ctxt):
            raise ValueError(f"value {_value} is not of type {self.referencedType}!")

        if _value is None:
            self.referenceName = "NULL"
            self._mangledReferenceName = "NULL"
        else:
            self.referenceName = _value
            self._mangledReferenceName = ctxt._mangle(_value)

    def __eq__(self, other):
        if not (isinstance(self, type(other)) and hasattr(other, "referenceName")):
            return False

        return self.referenceName == other.referenceName

    def __repr__(self):
        return f"{self._mangledReferenceName}"


class StructClass(metaclass = StructType):

    __slots__ = ["value"]
    typeName: str = None
    typeWidth: int = None
    structTypeDict: Dict[str, Union[ImmediateType, StructType, PointerType]] = None

    @classmethod
    def _setDict(cls, other, ctxt = None):
        _other = copy.deepcopy(other)

        for key, value in other.items():
            if not cls._compareType(other, key):
                _other[key] = cls.structTypeDict[key](other[key], ctxt)
            else:
                _other[key] = other[key]

        return _other

    @classmethod
    def _compareType(cls, other, key):
        if not (key in cls.structTypeDict):
            return False
        if not (isinstance(other[key], cls.structTypeDict[key])):
            return False
        return True

    @classmethod
    def checkValue(cls, other: Union[Dict, StructClass], ctxt = None):
        for key, value in other.items():
            if not cls.structTypeDict[key]._checkValue(value, ctxt):
                return False

        return True

    @classmethod
    def _checkValue(cls, _other: Union[Dict, StructClass], ctxt = None):

        if issubclass(type(_other), StructClass):
            other = _other.value
        else:
            other = _other

        if not hasattr(other, "keys"):
            return False

        if set(other.keys()) != set(cls.structTypeDict.keys()):
            return False

        return cls.checkValue(other, ctxt)

    def __init__(self, structDict: Dict[str, Union[ImmediateClass, PointerClass, StructClass]], ctxt = None):
        if isinstance(structDict, str):
            structDict = ctxt.lookup(structDict).structDict.value

        if not self._checkValue(structDict, ctxt):
            raise Exception(f"Can't assign {structDict} to {type(self)}!")

        self.value = self._setDict(structDict, ctxt)

    def __eq__(self, other):

        if not (hasattr(other, 'typeWidth') and hasattr(other, 'typeName') and hasattr(other, "value")):
            return False
        if any([not key in other.value.keys() for key in self.value.keys()]):
            return False
        return all([self.value[key] == other.value[key] for key in self.value.keys()])

    def __repr__(self):
        _repr = "{"
        pairs = []
        for key, value in self.value.items():
            pairs.append(f".{key} = {str(value)}")
        _repr += (", ").join(pairs)
        _repr += "}"
        return _repr

    def _typeDefRepr(self):
        _repr = "{"
        pairs = []
        for key, value in self.value.items():
            pairs.append(f"{value.typeName} {key}")
        _repr += ("; ").join(pairs)
        _repr += ";}"
        return _repr


def Pointer(dataType: Union[Union[ImmediateClass, PointerClass, StructClass], Union[ImmediateType, StructType,
                                                                                    PointerType]],
            pointerWidth = 32):

    if issubclass(type(dataType), (ImmediateClass, PointerClass, StructClass)):
        typeName = dataType.__class__.__name__
        ptrName = typeName + _POINTERSYMBOL

        return PointerType(ptrName, (PointerClass,), {
            "typeName": ptrName,
            "typeWidth": pointerWidth,
            "referencedType": dataType.__class__
        })(dataType)

    elif issubclass(type(dataType), (ImmediateType, PointerType, StructType)):
        typeName = dataType.typeName
        ptrName = typeName + _POINTERSYMBOL

        return PointerType(ptrName, (PointerClass,), {
            "typeName": ptrName,
            "typeWidth": pointerWidth,
            "referencedType": dataType
        })

    else:
        raise Exception(f"Can't create pointer to {dataType}!")


def Struct(typeName: str, structTypeDict: Dict[str:Union[ImmediateClass, PointerClass, StructClass]]):
    width = 0
    for _type in structTypeDict.values():
        width += _type.typeWidth
    return StructType(typeName, (StructClass,), {
        "typeName": typeName,
        "typeWidth": width,
        "structTypeDict": structTypeDict
    })


class HelperTypes(DataTypeCollection):
    void = IntegerType(32, True)
