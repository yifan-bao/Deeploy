# ----------------------------------------------------------------------
#
# File: Future.py
#
# Last edited: 07.06.2023
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

from typing import Union

from Deeploy.AbstractDataTypes import ImmediateClass, Pointer, PointerClass, PointerType, StructClass
from Deeploy.DeeployTypes import StructBuffer


class FutureType(PointerType):

    def __new__(cls, name, bases, namespace):
        retCls = super().__new__(cls, name, bases, namespace)

        assert "typeName" in namespace.keys(), "Missing typeName in future type constructor!"
        assert "typeWidth" in namespace.keys(), "Missing typeWidth in future type constructor!"
        assert "stateReferenceType" in namespace.keys(), "Missing stateReference in future type constructor!"
        assert "dispatchCheckTemplate" in namespace.keys(), "Missing dispatchCheckTemplate in future type constructor!"
        assert "resolveCheckTemplate" in namespace.keys(), "Missing resolveCheckTemplate in future type constructor!"

        return retCls

    def __instancecheck__(cls, other):
        if not super().__instancecheck__(other):
            return False

        if not (hasattr(other, "stateReferenceType")):
            return False

        return cls.stateReferenceType == other.stateReferenceType


class FutureClass(PointerClass, metaclass = FutureType):

    __slots__ = ['value', 'stateReference']
    typeName: str = None
    typeWidth: int = None
    referencedType: PointerClass = None
    stateReferenceType: PointerClass = None
    resolveCheckTemplate = None
    dispatchCheckTemplate = None

    def assignStateReference(self, stateReference: StructBuffer, ctxt: None):
        if self.stateReferenceType._checkValue(stateReference.structDict, ctxt):
            self.stateReference = stateReference
        else:
            raise Exception(f"Can't assign {stateReference} to {self}!")

    def _nodeRep(self):
        return {"stateReference": self.stateReference.name}


def Future(pointer: PointerClass,
           stateReferenceType: Union[ImmediateClass, PointerClass, StructClass, FutureClass],
           resolveCheckTemplate,
           dispatchCheckTemplate = None):
    width = pointer.typeWidth
    return FutureType(
        pointer.typeName, (FutureClass,), {
            "typeName": pointer.typeName,
            "typeWidth": width,
            "stateReferenceType": stateReferenceType,
            "referencedType": pointer.referencedType,
            "resolveCheckTemplate": resolveCheckTemplate,
            "dispatchCheckTemplate": dispatchCheckTemplate
        })
