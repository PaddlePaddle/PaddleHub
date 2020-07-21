# coding=utf-8
from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
import paddle.fluid as fluid


def face_detector():
    _319 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)
    _322 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=-1)
    _323 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=2)
    _333 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)
    _336 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=-1)
    _337 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=4)
    _365 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)
    _368 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=-1)
    _369 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=2)
    _379 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)
    _382 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=-1)
    _383 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=4)
    _405 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)
    _408 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=-1)
    _409 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=2)
    _419 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)
    _422 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=-1)
    _423 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=4)
    _437 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)
    _440 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=-1)
    _441 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=2)
    _449 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)
    _452 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=-1)
    _453 = fluid.layers.fill_constant(shape=[1], dtype='int32', value=4)
    _463 = fluid.layers.fill_constant(
        shape=[1], dtype='float32', value=0.10000000149011612)
    _465 = fluid.layers.create_parameter(
        dtype='float32',
        shape=[1, 17640, 2],
        name='_465',
        attr='_465',
        default_initializer=Constant(0.0))
    _467 = fluid.layers.create_parameter(
        dtype='float32',
        shape=[1, 17640, 2],
        name='_467',
        attr='_467',
        default_initializer=Constant(0.0))
    _470 = fluid.layers.fill_constant(
        shape=[1], dtype='float32', value=0.20000000298023224)
    _473 = fluid.layers.create_parameter(
        dtype='float32',
        shape=[1, 17640, 2],
        name='_473',
        attr='_473',
        default_initializer=Constant(0.0))
    _478 = fluid.layers.fill_constant(shape=[1], dtype='float32', value=2.0)
    _483 = fluid.layers.fill_constant(shape=[1], dtype='float32', value=2.0)
    _input = fluid.layers.data(
        dtype='float32',
        shape=[1, 3, 480, 640],
        name='_input',
        append_batch_size=False)
    _325 = fluid.layers.assign(_322)
    _326 = fluid.layers.assign(_323)
    _339 = fluid.layers.assign(_336)
    _340 = fluid.layers.assign(_337)
    _371 = fluid.layers.assign(_368)
    _372 = fluid.layers.assign(_369)
    _385 = fluid.layers.assign(_382)
    _386 = fluid.layers.assign(_383)
    _411 = fluid.layers.assign(_408)
    _412 = fluid.layers.assign(_409)
    _425 = fluid.layers.assign(_422)
    _426 = fluid.layers.assign(_423)
    _443 = fluid.layers.assign(_440)
    _444 = fluid.layers.assign(_441)
    _455 = fluid.layers.assign(_452)
    _456 = fluid.layers.assign(_453)
    _245 = fluid.layers.conv2d(
        _input,
        num_filters=16,
        filter_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_0_0_weight',
        name='_245',
        bias_attr=False)
    _246 = fluid.layers.batch_norm(
        _245,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_0_1_weight',
        bias_attr='_base_net_0_1_bias',
        moving_mean_name='_base_net_0_1_running_mean',
        moving_variance_name='_base_net_0_1_running_var',
        use_global_stats=False,
        name='_246')
    _247 = fluid.layers.relu(_246, name='_247')
    _248 = fluid.layers.conv2d(
        _247,
        num_filters=16,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=16,
        param_attr='_base_net_1_0_weight',
        name='_248',
        bias_attr=False)
    _249 = fluid.layers.batch_norm(
        _248,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_1_1_weight',
        bias_attr='_base_net_1_1_bias',
        moving_mean_name='_base_net_1_1_running_mean',
        moving_variance_name='_base_net_1_1_running_var',
        use_global_stats=False,
        name='_249')
    _250 = fluid.layers.relu(_249, name='_250')
    _251 = fluid.layers.conv2d(
        _250,
        num_filters=32,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_1_3_weight',
        name='_251',
        bias_attr=False)
    _252 = fluid.layers.batch_norm(
        _251,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_1_4_weight',
        bias_attr='_base_net_1_4_bias',
        moving_mean_name='_base_net_1_4_running_mean',
        moving_variance_name='_base_net_1_4_running_var',
        use_global_stats=False,
        name='_252')
    _253 = fluid.layers.relu(_252, name='_253')
    _254 = fluid.layers.conv2d(
        _253,
        num_filters=32,
        filter_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1],
        dilation=[1, 1],
        groups=32,
        param_attr='_base_net_2_0_weight',
        name='_254',
        bias_attr=False)
    _255 = fluid.layers.batch_norm(
        _254,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_2_1_weight',
        bias_attr='_base_net_2_1_bias',
        moving_mean_name='_base_net_2_1_running_mean',
        moving_variance_name='_base_net_2_1_running_var',
        use_global_stats=False,
        name='_255')
    _256 = fluid.layers.relu(_255, name='_256')
    _257 = fluid.layers.conv2d(
        _256,
        num_filters=32,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_2_3_weight',
        name='_257',
        bias_attr=False)
    _258 = fluid.layers.batch_norm(
        _257,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_2_4_weight',
        bias_attr='_base_net_2_4_bias',
        moving_mean_name='_base_net_2_4_running_mean',
        moving_variance_name='_base_net_2_4_running_var',
        use_global_stats=False,
        name='_258')
    _259 = fluid.layers.relu(_258, name='_259')
    _260 = fluid.layers.conv2d(
        _259,
        num_filters=32,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=32,
        param_attr='_base_net_3_0_weight',
        name='_260',
        bias_attr=False)
    _261 = fluid.layers.batch_norm(
        _260,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_3_1_weight',
        bias_attr='_base_net_3_1_bias',
        moving_mean_name='_base_net_3_1_running_mean',
        moving_variance_name='_base_net_3_1_running_var',
        use_global_stats=False,
        name='_261')
    _262 = fluid.layers.relu(_261, name='_262')
    _263 = fluid.layers.conv2d(
        _262,
        num_filters=32,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_3_3_weight',
        name='_263',
        bias_attr=False)
    _264 = fluid.layers.batch_norm(
        _263,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_3_4_weight',
        bias_attr='_base_net_3_4_bias',
        moving_mean_name='_base_net_3_4_running_mean',
        moving_variance_name='_base_net_3_4_running_var',
        use_global_stats=False,
        name='_264')
    _265 = fluid.layers.relu(_264, name='_265')
    _266 = fluid.layers.conv2d(
        _265,
        num_filters=32,
        filter_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1],
        dilation=[1, 1],
        groups=32,
        param_attr='_base_net_4_0_weight',
        name='_266',
        bias_attr=False)
    _267 = fluid.layers.batch_norm(
        _266,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_4_1_weight',
        bias_attr='_base_net_4_1_bias',
        moving_mean_name='_base_net_4_1_running_mean',
        moving_variance_name='_base_net_4_1_running_var',
        use_global_stats=False,
        name='_267')
    _268 = fluid.layers.relu(_267, name='_268')
    _269 = fluid.layers.conv2d(
        _268,
        num_filters=64,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_4_3_weight',
        name='_269',
        bias_attr=False)
    _270 = fluid.layers.batch_norm(
        _269,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_4_4_weight',
        bias_attr='_base_net_4_4_bias',
        moving_mean_name='_base_net_4_4_running_mean',
        moving_variance_name='_base_net_4_4_running_var',
        use_global_stats=False,
        name='_270')
    _271 = fluid.layers.relu(_270, name='_271')
    _272 = fluid.layers.conv2d(
        _271,
        num_filters=64,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=64,
        param_attr='_base_net_5_0_weight',
        name='_272',
        bias_attr=False)
    _273 = fluid.layers.batch_norm(
        _272,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_5_1_weight',
        bias_attr='_base_net_5_1_bias',
        moving_mean_name='_base_net_5_1_running_mean',
        moving_variance_name='_base_net_5_1_running_var',
        use_global_stats=False,
        name='_273')
    _274 = fluid.layers.relu(_273, name='_274')
    _275 = fluid.layers.conv2d(
        _274,
        num_filters=64,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_5_3_weight',
        name='_275',
        bias_attr=False)
    _276 = fluid.layers.batch_norm(
        _275,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_5_4_weight',
        bias_attr='_base_net_5_4_bias',
        moving_mean_name='_base_net_5_4_running_mean',
        moving_variance_name='_base_net_5_4_running_var',
        use_global_stats=False,
        name='_276')
    _277 = fluid.layers.relu(_276, name='_277')
    _278 = fluid.layers.conv2d(
        _277,
        num_filters=64,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=64,
        param_attr='_base_net_6_0_weight',
        name='_278',
        bias_attr=False)
    _279 = fluid.layers.batch_norm(
        _278,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_6_1_weight',
        bias_attr='_base_net_6_1_bias',
        moving_mean_name='_base_net_6_1_running_mean',
        moving_variance_name='_base_net_6_1_running_var',
        use_global_stats=False,
        name='_279')
    _280 = fluid.layers.relu(_279, name='_280')
    _281 = fluid.layers.conv2d(
        _280,
        num_filters=64,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_6_3_weight',
        name='_281',
        bias_attr=False)
    _282 = fluid.layers.batch_norm(
        _281,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_6_4_weight',
        bias_attr='_base_net_6_4_bias',
        moving_mean_name='_base_net_6_4_running_mean',
        moving_variance_name='_base_net_6_4_running_var',
        use_global_stats=False,
        name='_282')
    _283 = fluid.layers.relu(_282, name='_283')
    _284 = fluid.layers.conv2d(
        _283,
        num_filters=8,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_7_branch0_0_conv_weight',
        name='_284',
        bias_attr=False)
    _291 = fluid.layers.conv2d(
        _283,
        num_filters=8,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_7_branch1_0_conv_weight',
        name='_291',
        bias_attr=False)
    _298 = fluid.layers.conv2d(
        _283,
        num_filters=8,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_7_branch2_0_conv_weight',
        name='_298',
        bias_attr=False)
    _311 = fluid.layers.conv2d(
        _283,
        num_filters=64,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_7_shortcut_conv_weight',
        name='_311',
        bias_attr=False)
    _285 = fluid.layers.batch_norm(
        _284,
        momentum=0.9900000095367432,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_7_branch0_0_bn_weight',
        bias_attr='_base_net_7_branch0_0_bn_bias',
        moving_mean_name='_base_net_7_branch0_0_bn_running_mean',
        moving_variance_name='_base_net_7_branch0_0_bn_running_var',
        use_global_stats=False,
        name='_285')
    _292 = fluid.layers.batch_norm(
        _291,
        momentum=0.9900000095367432,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_7_branch1_0_bn_weight',
        bias_attr='_base_net_7_branch1_0_bn_bias',
        moving_mean_name='_base_net_7_branch1_0_bn_running_mean',
        moving_variance_name='_base_net_7_branch1_0_bn_running_var',
        use_global_stats=False,
        name='_292')
    _299 = fluid.layers.batch_norm(
        _298,
        momentum=0.9900000095367432,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_7_branch2_0_bn_weight',
        bias_attr='_base_net_7_branch2_0_bn_bias',
        moving_mean_name='_base_net_7_branch2_0_bn_running_mean',
        moving_variance_name='_base_net_7_branch2_0_bn_running_var',
        use_global_stats=False,
        name='_299')
    _312 = fluid.layers.batch_norm(
        _311,
        momentum=0.9900000095367432,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_7_shortcut_bn_weight',
        bias_attr='_base_net_7_shortcut_bn_bias',
        moving_mean_name='_base_net_7_shortcut_bn_running_mean',
        moving_variance_name='_base_net_7_shortcut_bn_running_var',
        use_global_stats=False,
        name='_312')
    _286 = fluid.layers.conv2d(
        _285,
        num_filters=16,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_7_branch0_1_conv_weight',
        name='_286',
        bias_attr=False)
    _293 = fluid.layers.conv2d(
        _292,
        num_filters=16,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_7_branch1_1_conv_weight',
        name='_293',
        bias_attr=False)
    _300 = fluid.layers.conv2d(
        _299,
        num_filters=12,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_7_branch2_1_conv_weight',
        name='_300',
        bias_attr=False)
    _287 = fluid.layers.batch_norm(
        _286,
        momentum=0.9900000095367432,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_7_branch0_1_bn_weight',
        bias_attr='_base_net_7_branch0_1_bn_bias',
        moving_mean_name='_base_net_7_branch0_1_bn_running_mean',
        moving_variance_name='_base_net_7_branch0_1_bn_running_var',
        use_global_stats=False,
        name='_287')
    _294 = fluid.layers.batch_norm(
        _293,
        momentum=0.9900000095367432,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_7_branch1_1_bn_weight',
        bias_attr='_base_net_7_branch1_1_bn_bias',
        moving_mean_name='_base_net_7_branch1_1_bn_running_mean',
        moving_variance_name='_base_net_7_branch1_1_bn_running_var',
        use_global_stats=False,
        name='_294')
    _301 = fluid.layers.batch_norm(
        _300,
        momentum=0.9900000095367432,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_7_branch2_1_bn_weight',
        bias_attr='_base_net_7_branch2_1_bn_bias',
        moving_mean_name='_base_net_7_branch2_1_bn_running_mean',
        moving_variance_name='_base_net_7_branch2_1_bn_running_var',
        use_global_stats=False,
        name='_301')
    _288 = fluid.layers.relu(_287, name='_288')
    _295 = fluid.layers.relu(_294, name='_295')
    _302 = fluid.layers.relu(_301, name='_302')
    _289 = fluid.layers.conv2d(
        _288,
        num_filters=16,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[2, 2],
        dilation=[2, 2],
        groups=1,
        param_attr='_base_net_7_branch0_2_conv_weight',
        name='_289',
        bias_attr=False)
    _296 = fluid.layers.conv2d(
        _295,
        num_filters=16,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[3, 3],
        dilation=[3, 3],
        groups=1,
        param_attr='_base_net_7_branch1_2_conv_weight',
        name='_296',
        bias_attr=False)
    _303 = fluid.layers.conv2d(
        _302,
        num_filters=16,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_7_branch2_2_conv_weight',
        name='_303',
        bias_attr=False)
    _290 = fluid.layers.batch_norm(
        _289,
        momentum=0.9900000095367432,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_7_branch0_2_bn_weight',
        bias_attr='_base_net_7_branch0_2_bn_bias',
        moving_mean_name='_base_net_7_branch0_2_bn_running_mean',
        moving_variance_name='_base_net_7_branch0_2_bn_running_var',
        use_global_stats=False,
        name='_290')
    _297 = fluid.layers.batch_norm(
        _296,
        momentum=0.9900000095367432,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_7_branch1_2_bn_weight',
        bias_attr='_base_net_7_branch1_2_bn_bias',
        moving_mean_name='_base_net_7_branch1_2_bn_running_mean',
        moving_variance_name='_base_net_7_branch1_2_bn_running_var',
        use_global_stats=False,
        name='_297')
    _304 = fluid.layers.batch_norm(
        _303,
        momentum=0.9900000095367432,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_7_branch2_2_bn_weight',
        bias_attr='_base_net_7_branch2_2_bn_bias',
        moving_mean_name='_base_net_7_branch2_2_bn_running_mean',
        moving_variance_name='_base_net_7_branch2_2_bn_running_var',
        use_global_stats=False,
        name='_304')
    _305 = fluid.layers.relu(_304, name='_305')
    _306 = fluid.layers.conv2d(
        _305,
        num_filters=16,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[5, 5],
        dilation=[5, 5],
        groups=1,
        param_attr='_base_net_7_branch2_3_conv_weight',
        name='_306',
        bias_attr=False)
    _307 = fluid.layers.batch_norm(
        _306,
        momentum=0.9900000095367432,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_7_branch2_3_bn_weight',
        bias_attr='_base_net_7_branch2_3_bn_bias',
        moving_mean_name='_base_net_7_branch2_3_bn_running_mean',
        moving_variance_name='_base_net_7_branch2_3_bn_running_var',
        use_global_stats=False,
        name='_307')
    _308 = fluid.layers.concat([_290, _297, _307], axis=1)
    _309 = fluid.layers.conv2d(
        _308,
        num_filters=64,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_7_ConvLinear_conv_weight',
        name='_309',
        bias_attr=False)
    _310 = fluid.layers.batch_norm(
        _309,
        momentum=0.9900000095367432,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_7_ConvLinear_bn_weight',
        bias_attr='_base_net_7_ConvLinear_bn_bias',
        moving_mean_name='_base_net_7_ConvLinear_bn_running_mean',
        moving_variance_name='_base_net_7_ConvLinear_bn_running_var',
        use_global_stats=False,
        name='_310')
    _313 = fluid.layers.elementwise_add(x=_310, y=_312, name='_313')
    _314 = fluid.layers.relu(_313, name='_314')
    _315 = fluid.layers.conv2d(
        _314,
        num_filters=64,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=64,
        param_attr='_classification_headers_0_0_weight',
        name='_315',
        bias_attr='_classification_headers_0_0_bias')
    _329 = fluid.layers.conv2d(
        _314,
        num_filters=64,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=64,
        param_attr='_regression_headers_0_0_weight',
        name='_329',
        bias_attr='_regression_headers_0_0_bias')
    _343 = fluid.layers.conv2d(
        _314,
        num_filters=64,
        filter_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1],
        dilation=[1, 1],
        groups=64,
        param_attr='_base_net_8_0_weight',
        name='_343',
        bias_attr=False)
    _316 = fluid.layers.relu(_315, name='_316')
    _330 = fluid.layers.relu(_329, name='_330')
    _344 = fluid.layers.batch_norm(
        _343,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_8_1_weight',
        bias_attr='_base_net_8_1_bias',
        moving_mean_name='_base_net_8_1_running_mean',
        moving_variance_name='_base_net_8_1_running_var',
        use_global_stats=False,
        name='_344')
    _317 = fluid.layers.conv2d(
        _316,
        num_filters=6,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_classification_headers_0_2_weight',
        name='_317',
        bias_attr='_classification_headers_0_2_bias')
    _331 = fluid.layers.conv2d(
        _330,
        num_filters=12,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_regression_headers_0_2_weight',
        name='_331',
        bias_attr='_regression_headers_0_2_bias')
    _345 = fluid.layers.relu(_344, name='_345')
    _318 = fluid.layers.transpose(_317, perm=[0, 2, 3, 1], name='_318')
    _332 = fluid.layers.transpose(_331, perm=[0, 2, 3, 1], name='_332')
    _346 = fluid.layers.conv2d(
        _345,
        num_filters=128,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_8_3_weight',
        name='_346',
        bias_attr=False)
    _320 = fluid.layers.shape(_318)
    _334 = fluid.layers.shape(_332)
    _347 = fluid.layers.batch_norm(
        _346,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_8_4_weight',
        bias_attr='_base_net_8_4_bias',
        moving_mean_name='_base_net_8_4_running_mean',
        moving_variance_name='_base_net_8_4_running_var',
        use_global_stats=False,
        name='_347')
    _321 = fluid.layers.gather(input=_320, index=_319)
    _335 = fluid.layers.gather(input=_334, index=_333)
    _348 = fluid.layers.relu(_347, name='_348')
    _324 = fluid.layers.assign(_321)
    _338 = fluid.layers.assign(_335)
    _349 = fluid.layers.conv2d(
        _348,
        num_filters=128,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=128,
        param_attr='_base_net_9_0_weight',
        name='_349',
        bias_attr=False)
    _327 = fluid.layers.concat([_324, _325, _326], axis=0)
    _341 = fluid.layers.concat([_338, _339, _340], axis=0)
    _350 = fluid.layers.batch_norm(
        _349,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_9_1_weight',
        bias_attr='_base_net_9_1_bias',
        moving_mean_name='_base_net_9_1_running_mean',
        moving_variance_name='_base_net_9_1_running_var',
        use_global_stats=False,
        name='_350')
    _327_cast = fluid.layers.cast(_327, dtype='int32')
    _328 = fluid.layers.reshape(
        _318, name='_328', actual_shape=_327_cast, shape=[1, -1, 2])
    _341_cast = fluid.layers.cast(_341, dtype='int32')
    _342 = fluid.layers.reshape(
        _332, name='_342', actual_shape=_341_cast, shape=[1, -1, 4])
    _351 = fluid.layers.relu(_350, name='_351')
    _352 = fluid.layers.conv2d(
        _351,
        num_filters=128,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_9_3_weight',
        name='_352',
        bias_attr=False)
    _353 = fluid.layers.batch_norm(
        _352,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_9_4_weight',
        bias_attr='_base_net_9_4_bias',
        moving_mean_name='_base_net_9_4_running_mean',
        moving_variance_name='_base_net_9_4_running_var',
        use_global_stats=False,
        name='_353')
    _354 = fluid.layers.relu(_353, name='_354')
    _355 = fluid.layers.conv2d(
        _354,
        num_filters=128,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=128,
        param_attr='_base_net_10_0_weight',
        name='_355',
        bias_attr=False)
    _356 = fluid.layers.batch_norm(
        _355,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_10_1_weight',
        bias_attr='_base_net_10_1_bias',
        moving_mean_name='_base_net_10_1_running_mean',
        moving_variance_name='_base_net_10_1_running_var',
        use_global_stats=False,
        name='_356')
    _357 = fluid.layers.relu(_356, name='_357')
    _358 = fluid.layers.conv2d(
        _357,
        num_filters=128,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_10_3_weight',
        name='_358',
        bias_attr=False)
    _359 = fluid.layers.batch_norm(
        _358,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_10_4_weight',
        bias_attr='_base_net_10_4_bias',
        moving_mean_name='_base_net_10_4_running_mean',
        moving_variance_name='_base_net_10_4_running_var',
        use_global_stats=False,
        name='_359')
    _360 = fluid.layers.relu(_359, name='_360')
    _361 = fluid.layers.conv2d(
        _360,
        num_filters=128,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=128,
        param_attr='_classification_headers_1_0_weight',
        name='_361',
        bias_attr='_classification_headers_1_0_bias')
    _375 = fluid.layers.conv2d(
        _360,
        num_filters=128,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=128,
        param_attr='_regression_headers_1_0_weight',
        name='_375',
        bias_attr='_regression_headers_1_0_bias')
    _389 = fluid.layers.conv2d(
        _360,
        num_filters=128,
        filter_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1],
        dilation=[1, 1],
        groups=128,
        param_attr='_base_net_11_0_weight',
        name='_389',
        bias_attr=False)
    _362 = fluid.layers.relu(_361, name='_362')
    _376 = fluid.layers.relu(_375, name='_376')
    _390 = fluid.layers.batch_norm(
        _389,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_11_1_weight',
        bias_attr='_base_net_11_1_bias',
        moving_mean_name='_base_net_11_1_running_mean',
        moving_variance_name='_base_net_11_1_running_var',
        use_global_stats=False,
        name='_390')
    _363 = fluid.layers.conv2d(
        _362,
        num_filters=4,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_classification_headers_1_2_weight',
        name='_363',
        bias_attr='_classification_headers_1_2_bias')
    _377 = fluid.layers.conv2d(
        _376,
        num_filters=8,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_regression_headers_1_2_weight',
        name='_377',
        bias_attr='_regression_headers_1_2_bias')
    _391 = fluid.layers.relu(_390, name='_391')
    _364 = fluid.layers.transpose(_363, perm=[0, 2, 3, 1], name='_364')
    _378 = fluid.layers.transpose(_377, perm=[0, 2, 3, 1], name='_378')
    _392 = fluid.layers.conv2d(
        _391,
        num_filters=256,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_11_3_weight',
        name='_392',
        bias_attr=False)
    _366 = fluid.layers.shape(_364)
    _380 = fluid.layers.shape(_378)
    _393 = fluid.layers.batch_norm(
        _392,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_11_4_weight',
        bias_attr='_base_net_11_4_bias',
        moving_mean_name='_base_net_11_4_running_mean',
        moving_variance_name='_base_net_11_4_running_var',
        use_global_stats=False,
        name='_393')
    _367 = fluid.layers.gather(input=_366, index=_365)
    _381 = fluid.layers.gather(input=_380, index=_379)
    _394 = fluid.layers.relu(_393, name='_394')
    _370 = fluid.layers.assign(_367)
    _384 = fluid.layers.assign(_381)
    _395 = fluid.layers.conv2d(
        _394,
        num_filters=256,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=256,
        param_attr='_base_net_12_0_weight',
        name='_395',
        bias_attr=False)
    _373 = fluid.layers.concat([_370, _371, _372], axis=0)
    _387 = fluid.layers.concat([_384, _385, _386], axis=0)
    _396 = fluid.layers.batch_norm(
        _395,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_12_1_weight',
        bias_attr='_base_net_12_1_bias',
        moving_mean_name='_base_net_12_1_running_mean',
        moving_variance_name='_base_net_12_1_running_var',
        use_global_stats=False,
        name='_396')
    _373_cast = fluid.layers.cast(_373, dtype='int32')
    _374 = fluid.layers.reshape(
        _364, name='_374', actual_shape=_373_cast, shape=[1, -1, 2])
    _387_cast = fluid.layers.cast(_387, dtype='int32')
    _388 = fluid.layers.reshape(
        _378, name='_388', actual_shape=_387_cast, shape=[1, -1, 4])
    _397 = fluid.layers.relu(_396, name='_397')
    _398 = fluid.layers.conv2d(
        _397,
        num_filters=256,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_base_net_12_3_weight',
        name='_398',
        bias_attr=False)
    _399 = fluid.layers.batch_norm(
        _398,
        momentum=0.8999999761581421,
        epsilon=9.999999747378752e-06,
        data_layout='NCHW',
        is_test=True,
        param_attr='_base_net_12_4_weight',
        bias_attr='_base_net_12_4_bias',
        moving_mean_name='_base_net_12_4_running_mean',
        moving_variance_name='_base_net_12_4_running_var',
        use_global_stats=False,
        name='_399')
    _400 = fluid.layers.relu(_399, name='_400')
    _401 = fluid.layers.conv2d(
        _400,
        num_filters=256,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=256,
        param_attr='_classification_headers_2_0_weight',
        name='_401',
        bias_attr='_classification_headers_2_0_bias')
    _415 = fluid.layers.conv2d(
        _400,
        num_filters=256,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=256,
        param_attr='_regression_headers_2_0_weight',
        name='_415',
        bias_attr='_regression_headers_2_0_bias')
    _429 = fluid.layers.conv2d(
        _400,
        num_filters=64,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_extras_0_0_weight',
        name='_429',
        bias_attr='_extras_0_0_bias')
    _402 = fluid.layers.relu(_401, name='_402')
    _416 = fluid.layers.relu(_415, name='_416')
    _430 = fluid.layers.relu(_429, name='_430')
    _403 = fluid.layers.conv2d(
        _402,
        num_filters=4,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_classification_headers_2_2_weight',
        name='_403',
        bias_attr='_classification_headers_2_2_bias')
    _417 = fluid.layers.conv2d(
        _416,
        num_filters=8,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_regression_headers_2_2_weight',
        name='_417',
        bias_attr='_regression_headers_2_2_bias')
    _431 = fluid.layers.conv2d(
        _430,
        num_filters=64,
        filter_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1],
        dilation=[1, 1],
        groups=64,
        param_attr='_extras_0_2_0_weight',
        name='_431',
        bias_attr='_extras_0_2_0_bias')
    _404 = fluid.layers.transpose(_403, perm=[0, 2, 3, 1], name='_404')
    _418 = fluid.layers.transpose(_417, perm=[0, 2, 3, 1], name='_418')
    _432 = fluid.layers.relu(_431, name='_432')
    _406 = fluid.layers.shape(_404)
    _420 = fluid.layers.shape(_418)
    _433 = fluid.layers.conv2d(
        _432,
        num_filters=256,
        filter_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        param_attr='_extras_0_2_2_weight',
        name='_433',
        bias_attr='_extras_0_2_2_bias')
    _407 = fluid.layers.gather(input=_406, index=_405)
    _421 = fluid.layers.gather(input=_420, index=_419)
    _434 = fluid.layers.relu(_433, name='_434')
    _410 = fluid.layers.assign(_407)
    _424 = fluid.layers.assign(_421)
    _435 = fluid.layers.conv2d(
        _434,
        num_filters=6,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=1,
        param_attr='_classification_headers_3_weight',
        name='_435',
        bias_attr='_classification_headers_3_bias')
    _447 = fluid.layers.conv2d(
        _434,
        num_filters=12,
        filter_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        groups=1,
        param_attr='_regression_headers_3_weight',
        name='_447',
        bias_attr='_regression_headers_3_bias')
    _413 = fluid.layers.concat([_410, _411, _412], axis=0)
    _427 = fluid.layers.concat([_424, _425, _426], axis=0)
    _436 = fluid.layers.transpose(_435, perm=[0, 2, 3, 1], name='_436')
    _448 = fluid.layers.transpose(_447, perm=[0, 2, 3, 1], name='_448')
    _413_cast = fluid.layers.cast(_413, dtype='int32')
    _414 = fluid.layers.reshape(
        _404, name='_414', actual_shape=_413_cast, shape=[1, -1, 2])
    _427_cast = fluid.layers.cast(_427, dtype='int32')
    _428 = fluid.layers.reshape(
        _418, name='_428', actual_shape=_427_cast, shape=[1, -1, 4])
    _438 = fluid.layers.shape(_436)
    _450 = fluid.layers.shape(_448)
    _439 = fluid.layers.gather(input=_438, index=_437)
    _451 = fluid.layers.gather(input=_450, index=_449)
    _442 = fluid.layers.assign(_439)
    _454 = fluid.layers.assign(_451)
    _445 = fluid.layers.concat([_442, _443, _444], axis=0)
    _457 = fluid.layers.concat([_454, _455, _456], axis=0)
    _445_cast = fluid.layers.cast(_445, dtype='int32')
    _446 = fluid.layers.reshape(
        _436, name='_446', actual_shape=_445_cast, shape=[1, -1, 2])
    _457_cast = fluid.layers.cast(_457, dtype='int32')
    _458 = fluid.layers.reshape(
        _448, name='_458', actual_shape=_457_cast, shape=[1, -1, 4])
    _459 = fluid.layers.concat([_328, _374, _414, _446], axis=1)
    _460 = fluid.layers.concat([_342, _388, _428, _458], axis=1)
    _scores = fluid.layers.softmax(_459, axis=2, name='_scores')
    _462 = fluid.layers.slice(_460, axes=[2], starts=[0], ends=[2])
    _469 = fluid.layers.slice(_460, axes=[2], starts=[2], ends=[4])
    _464 = fluid.layers.elementwise_mul(x=_462, y=_463, name='_464')
    _471 = fluid.layers.elementwise_mul(x=_469, y=_470, name='_471')
    _466 = fluid.layers.elementwise_mul(x=_464, y=_465, name='_466')
    _472 = fluid.layers.exp(_471, name='_472')
    _468 = fluid.layers.elementwise_add(x=_466, y=_467, name='_468')
    _474 = fluid.layers.elementwise_mul(x=_472, y=_473, name='_474')
    _475 = fluid.layers.concat([_468, _474], axis=2)
    _476 = fluid.layers.slice(_475, axes=[2], starts=[0], ends=[2])
    _477 = fluid.layers.slice(_475, axes=[2], starts=[2], ends=[4])
    _481 = fluid.layers.slice(_475, axes=[2], starts=[0], ends=[2])
    _482 = fluid.layers.slice(_475, axes=[2], starts=[2], ends=[4])
    _479 = fluid.layers.elementwise_div(x=_477, y=_478, name='_479')
    _484 = fluid.layers.elementwise_div(x=_482, y=_483, name='_484')
    _480 = fluid.layers.elementwise_sub(x=_476, y=_479, name='_480')
    _485 = fluid.layers.elementwise_add(x=_481, y=_484, name='_485')
    _boxes = fluid.layers.concat([_480, _485], axis=2)

    return [_input], [_scores, _boxes]


def run_net(param_dir="./"):
    import os
    inputs, outputs = face_detector()
    for i, out in enumerate(outputs):
        if isinstance(out, list):
            for out_part in out:
                outputs.append(out_part)
            del outputs[i]
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    def if_exist(var):
        b = os.path.exists(os.path.join(param_dir, var.name))
        return b

    fluid.io.load_vars(
        exe, param_dir, fluid.default_main_program(), predicate=if_exist)
