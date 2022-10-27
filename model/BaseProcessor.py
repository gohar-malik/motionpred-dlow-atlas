"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys

sys.path.append("/home/HwHiAiUser/Documents/lib")

import os 
from abc import abstractmethod

from atlas_utils.acl_resource import AclResource
from atlas_utils.acl_model import Model

class BaseProcessor:
    def __init__(self, params):
        # Initialize ACL Resources
        self._acl_resource = AclResource()
        self._acl_resource.init()
        self.model = Model(params['model_path'])

    @abstractmethod
    def preprocess(self):
        pass
        
    @abstractmethod    
    def postprocess(self):
        pass

    @abstractmethod
    def predict(self):
        pass

