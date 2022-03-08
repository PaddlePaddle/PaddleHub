# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np


# import scipy.io as
def frontalize(vertices):
    canonical_vertices = np.load('Data/uv-data/canonical_vertices.npy')

    vertices_homo = np.hstack((vertices, np.ones([vertices.shape[0], 1])))  #n x 4
    P = np.linalg.lstsq(vertices_homo, canonical_vertices)[0].T  # Affine matrix. 3 x 4
    front_vertices = vertices_homo.dot(P.T)

    return front_vertices
