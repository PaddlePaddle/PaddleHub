# -*- coding: UTF-8 -*-
################################################################################
#
#   Copyright (c) 2020  Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
#################################################################################
"""本文件定义全连接层"""

from paddle.fluid import dygraph
from paddle.fluid import initializer
from paddle.fluid import layers

from parser.nets import dropouts


class MLP(dygraph.Layer):
    """MLP"""

    def __init__(self, n_in, n_out, dropout=0):
        super(MLP, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.linear = dygraph.Linear(
            n_in,
            n_out,
            param_attr=initializer.Xavier(uniform=False),
            bias_attr=None,
        )
        self.dropout = dropouts.SharedDropout(p=dropout)

    def __repr__(self):
        """repr"""
        s = self.__class__.__name__ + '('
        s += f"n_in={self.n_in}, n_out={self.n_out}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"
        s += ')'

        return s

    def forward(self, x):
        """Forward network"""
        x = self.linear(x)
        x = layers.leaky_relu(x, alpha=0.1)
        x = self.dropout(x)

        return x
