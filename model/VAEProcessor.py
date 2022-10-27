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
import acl
import os
import cv2
import numpy as np
import sys
import time

sys.path.append(os.path.join(os.getcwd(), "model"))
from BaseProcessor import BaseProcessor

sys.path.append("/home/HwHiAiUser/Documents/lib")
from atlas_utils.resource_list import resource_list


class ModelProcessor(BaseProcessor):
    def __init__(self, params):
        self.params = params
        self.validate()
        super().__init__(params)
        
        self.batch_size = params["batch_size"]
        self.num_seeds = params["num_seeds"]
        self.nz = params["nz"]
        
        
    def validate(self):
        if not os.path.exists(self.params['model_path']):
            raise FileNotFoundError('Model Path not found, please check again.')
        if 'batch_size' not in self.params:
            raise Exception('Please specify batch_size for model in params')

    def release_acl(self):
        print("acl resource release all resource")
        resource_list.destroy()
        if self._acl_resource.stream:
            print("acl resource release stream")
            acl.rt.destroy_stream(self._acl_resource.stream)

        if self._acl_resource.context:
            print("acl resource release context")
            acl.rt.destroy_context(self._acl_resource.context)

        print("Reset acl device ", self._acl_resource.device_id)
        acl.rt.reset_device(self._acl_resource.device_id)
        acl.finalize()
        print("Release acl resource success")

    def predict(self, data, t_his, concat_hist):
        X, Z = self.preprocess(data, t_his)
        
        Y = self.model.execute([X, Z])
        # Y = np.zeros((100,1,48))
        if Y is None:
            return Y
        
        result = self.postprocess(X, Y, concat_hist)
        return result

    def preprocess(self, data, t_his):
        traj_np = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
        traj = np.transpose(traj_np, (1, 0, 2))
        X = traj[:t_his]
        X = np.tile(X, (1, self.batch_size * self.num_seeds, 1))
        Z = np.random.randn(X.shape[1], self.nz).astype("float32")
        
        # X = np.load("X.npy")
        # Z = np.load("Z.npy")
        
        return X, Z
        
    def postprocess(self, X, Y, concat_hist):
        if concat_hist:
            Y = np.concatenate((X, Y), axis=0)
    
        Y = np.transpose(Y, (1, 0, 2))
        if Y.shape[0] > 1:
            Y = Y.reshape(-1, self.batch_size, Y.shape[-2], Y.shape[-1])
        else:
            Y = Y[None, ...]
        return Y
