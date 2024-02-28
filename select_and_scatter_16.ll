HloModule a_inference_one_step_on_data_1486__.1059, input_output_alias={ {2}: (2, {}, may-alias), {3}: (3, {}, may-alias), {4}: (4, {}, may-alias), {5}: (5, {}, may-alias), {6}: (6, {}, may-alias), {7}: (7, {}, may-alias), {8}: (8, {}, may-alias), {9}: (9, {}, may-alias), {10}: (10, {}, may-alias), {11}: (11, {}, may-alias), {12}: (12, {}, may-alias), {13}: (14, {}, may-alias), {14}: (15, {}, may-alias), {15}: (16, {}, may-alias), {16}: (17, {}, may-alias), {17}: (18, {}, may-alias), {18}: (19, {}, may-alias), {19}: (20, {}, may-alias), {20}: (21, {}, may-alias), {21}: (22, {}, may-alias), {22}: (23, {}, may-alias), {23}: (24, {}, may-alias), {24}: (25, {}, may-alias) }, alias_passthrough_params=true, entry_computation_layout={(f32[768,96,96,64]{3,2,1,0}, s32[768]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=5*/f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=10*/f32[], f32[], f32[], f32[], s64[], /*index=15*/f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=20*/f32[64]{0}, f32[64]{0}, f32[64]{0}, s64[], f32[], /*index=25*/f32[])->(f32[], f32[], f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=5*/f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=10*/f32[], f32[], f32[], s64[], f32[64]{0}, /*index=15*/f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=20*/f32[64]{0}, f32[64]{0}, s64[], f32[], f32[])}

sequential_1_batch_normalization_1_moments_mean-reduction.65 {
  x.66 = f32[] parameter(0)
  y.67 = f32[] parameter(1)
  ROOT add.68 = f32[] add(x.66, y.67)
}

sequential_1_batch_normalization_1_moments_variance-reduction.88 {
  x.89 = f32[] parameter(0)
  y.90 = f32[] parameter(1)
  ROOT add.91 = f32[] add(x.89, y.90)
}

max_F16.170 {
  lhs.171 = f16[] parameter(0)
  rhs.172 = f16[] parameter(1)
  ROOT maximum.173 = f16[] maximum(lhs.171, rhs.172)
}

gradient_tape_sequential_1_batch_normalization_1_2_moments_Prod-reduction.220 {
  x.221 = s32[] parameter(0)
  y.222 = s32[] parameter(1)
  ROOT multiply.223 = s32[] multiply(x.221, y.222)
}

sequential_1_batch_normalization_1_2_moments_mean-reduction.243 {
  x.244 = f32[] parameter(0)
  y.245 = f32[] parameter(1)
  ROOT add.246 = f32[] add(x.244, y.245)
}

gradient_tape_sequential_1_batch_normalization_1_2_moments_Prod_1-reduction.300 {
  x.301 = s32[] parameter(0)
  y.302 = s32[] parameter(1)
  ROOT multiply.303 = s32[] multiply(x.301, y.302)
}

sequential_1_batch_normalization_1_2_moments_variance-reduction.310 {
  x.311 = f32[] parameter(0)
  y.312 = f32[] parameter(1)
  ROOT add.313 = f32[] add(x.311, y.312)
}

minmax_func.371 {
  lhs_value.372 = f16[] parameter(0)
  rhs_value.374 = f16[] parameter(2)
  compare.376 = pred[] compare(lhs_value.372, rhs_value.374), direction=GE
  select.377 = f16[] select(compare.376, lhs_value.372, rhs_value.374)
  compare.379 = pred[] compare(lhs_value.372, rhs_value.374), direction=EQ
  lhs_index.373 = s32[] parameter(1)
  rhs_index.375 = s32[] parameter(3)
  minimum.380 = s32[] minimum(lhs_index.373, rhs_index.375)
  select.378 = s32[] select(compare.376, lhs_index.373, rhs_index.375)
  select.381 = s32[] select(compare.379, minimum.380, select.378)
  ROOT tuple.382 = (f16[], s32[]) tuple(select.377, select.381)
} // minmax_func.371

Sum_1-reduction.396 {
  x.397 = f32[] parameter(0)
  y.398 = f32[] parameter(1)
  ROOT add.399 = f32[] add(x.397, y.398)
}

max_float_.435 {
  x.436 = f32[] parameter(0)
  y.437 = f32[] parameter(1)
  ROOT maximum.438 = f32[] maximum(x.436, y.437)
}

add_float_.445 {
  x.446 = f32[] parameter(0)
  y.447 = f32[] parameter(1)
  ROOT add.448 = f32[] add(x.446, y.447)
}

add_float_.464 {
  x.465 = f32[] parameter(0)
  y.466 = f32[] parameter(1)
  ROOT add.467 = f32[] add(x.465, y.466)
}

compile_loss_sparse_categorical_crossentropy_Sum-reduction.479 {
  x.480 = f32[] parameter(0)
  y.481 = f32[] parameter(1)
  ROOT add.482 = f32[] add(x.480, y.481)
}

gradient_tape_sequential_1_batch_normalization_1_2_batchnorm_add_1_Sum-reduction.534 {
  x.535 = f32[] parameter(0)
  y.536 = f32[] parameter(1)
  ROOT add.537 = f32[] add(x.535, y.536)
}

All_3-reduction.548 {
  x.549 = pred[] parameter(0)
  y.550 = pred[] parameter(1)
  ROOT and.551 = pred[] and(x.549, y.550)
}

gradient_tape_sequential_1_batch_normalization_1_2_batchnorm_mul_1_Sum-reduction.569 {
  x.570 = f32[] parameter(0)
  y.571 = f32[] parameter(1)
  ROOT add.572 = f32[] add(x.570, y.571)
}

max_F16.602 {
  lhs.603 = f16[] parameter(0)
  rhs.604 = f16[] parameter(1)
  ROOT maximum.605 = f16[] maximum(lhs.603, rhs.604)
}

ge_F16.608 {
  lhs.609 = f16[] parameter(0)
  rhs.610 = f16[] parameter(1)
  ROOT compare.611 = pred[] compare(lhs.609, rhs.610), direction=GE
}

add_F16.612 {
  lhs.613 = f16[] parameter(0)
  rhs.614 = f16[] parameter(1)
  ROOT add.615 = f16[] add(lhs.613, rhs.614)
}

gradient_tape_sequential_1_batch_normalization_1_batchnorm_add_1_Sum-reduction.621 {
  x.622 = f32[] parameter(0)
  y.623 = f32[] parameter(1)
  ROOT add.624 = f32[] add(x.622, y.623)
}

All_1-reduction.635 {
  x.636 = pred[] parameter(0)
  y.637 = pred[] parameter(1)
  ROOT and.638 = pred[] and(x.636, y.637)
}

gradient_tape_sequential_1_batch_normalization_1_batchnorm_mul_1_Sum-reduction.648 {
  x.649 = f32[] parameter(0)
  y.650 = f32[] parameter(1)
  ROOT add.651 = f32[] add(x.649, y.650)
}

All-reduction.665 {
  x.666 = pred[] parameter(0)
  y.667 = pred[] parameter(1)
  ROOT and.668 = pred[] and(x.666, y.667)
}

All_2-reduction.678 {
  x.679 = pred[] parameter(0)
  y.680 = pred[] parameter(1)
  ROOT and.681 = pred[] and(x.679, y.680)
}

All_4-reduction.692 {
  x.693 = pred[] parameter(0)
  y.694 = pred[] parameter(1)
  ROOT and.695 = pred[] and(x.693, y.694)
}

cond_cond_true_1399__.699 {
  arg_tuple.700 = (s64[], f32[]) parameter(0), metadata={op_name="XLA_Args"}
  get-tuple-element.701 = s64[] get-tuple-element(arg_tuple.700), index=0
  constant.703 = s64[] constant(0), metadata={op_type="AssignVariableOp" op_name="cond/cond/AssignVariableOp" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.706 = s64[] reshape(constant.703), metadata={op_name="XLA_Retvals"}
  copy.707 = s64[] copy(reshape.706), metadata={op_name="XLA_Retvals"}
  get-tuple-element.702 = f32[] get-tuple-element(arg_tuple.700), index=1
  constant.704 = f32[] constant(2), metadata={op_type="Mul" op_name="cond/cond/mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.705 = f32[] multiply(get-tuple-element.702, constant.704), metadata={op_type="Mul" op_name="cond/cond/mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.708 = f32[] reshape(multiply.705), metadata={op_name="XLA_Retvals"}
  copy.709 = f32[] copy(reshape.708), metadata={op_name="XLA_Retvals"}
  ROOT tuple.710 = (s64[], f32[]) tuple(copy.707, copy.709), metadata={op_name="XLA_Retvals"}
} // cond_cond_true_1399__.699

cond_cond_false_1400__.711 {
  arg_tuple.712 = (s64[], f32[]) parameter(0), metadata={op_name="XLA_Args"}
  get-tuple-element.713 = s64[] get-tuple-element(arg_tuple.712), index=0
  constant.715 = s64[] constant(1), metadata={op_type="AddV2" op_name="cond/cond/add" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.716 = s64[] add(get-tuple-element.713, constant.715), metadata={op_type="AddV2" op_name="cond/cond/add" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.717 = s64[] reshape(add.716), metadata={op_name="XLA_Retvals"}
  copy.718 = s64[] copy(reshape.717), metadata={op_name="XLA_Retvals"}
  get-tuple-element.714 = f32[] get-tuple-element(arg_tuple.712), index=1
  reshape.719 = f32[] reshape(get-tuple-element.714), metadata={op_name="XLA_Retvals"}
  copy.720 = f32[] copy(reshape.719), metadata={op_name="XLA_Retvals"}
  ROOT tuple.721 = (s64[], f32[]) tuple(copy.718, copy.720), metadata={op_name="XLA_Retvals"}
} // cond_cond_false_1400__.711

cond_true_1233_rearrange_0__.722 {
  arg_tuple.723 = (f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[], /*index=5*/f32[], s64[], f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=10*/f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=15*/f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, s64[]) parameter(0), metadata={op_name="XLA_Args"}
  get-tuple-element.743 = s64[] get-tuple-element(arg_tuple.723), index=19
  constant.802 = s64[] constant(1999), metadata={op_type="Equal" op_name="cond/Equal" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  compare.803 = pred[] compare(get-tuple-element.743, constant.802), direction=EQ, metadata={op_type="Equal" op_name="cond/Equal" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  get-tuple-element.728 = f32[] get-tuple-element(arg_tuple.723), index=4
  tuple.804 = (s64[], f32[]) tuple(get-tuple-element.743, get-tuple-element.728), metadata={op_type="If" op_name="cond/cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  conditional.805 = (s64[], f32[]) conditional(compare.803, tuple.804, tuple.804), true_computation=cond_cond_true_1399__.699, false_computation=cond_cond_false_1400__.711, metadata={op_type="If" op_name="cond/cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  get-tuple-element.806 = s64[] get-tuple-element(conditional.805), index=0, metadata={op_type="If" op_name="cond/cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  get-tuple-element.807 = f32[] get-tuple-element(conditional.805), index=1, metadata={op_type="If" op_name="cond/cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.893 = f32[] reshape(get-tuple-element.807), metadata={op_name="XLA_Retvals"}
  copy.894 = f32[] copy(reshape.893), metadata={op_name="XLA_Retvals"}
  get-tuple-element.729 = f32[] get-tuple-element(arg_tuple.723), index=5
  reshape.895 = f32[] reshape(get-tuple-element.729), metadata={op_name="XLA_Retvals"}
  copy.896 = f32[] copy(reshape.895), metadata={op_name="XLA_Retvals"}
  get-tuple-element.730 = s64[] get-tuple-element(arg_tuple.723), index=6
  constant.800 = s64[] constant(1), metadata={op_type="AddV2" op_name="cond/adam/add_8" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.801 = s64[] add(get-tuple-element.730, constant.800), metadata={op_type="AddV2" op_name="cond/adam/add_8" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.897 = s64[] reshape(add.801), metadata={op_name="XLA_Retvals"}
  copy.898 = s64[] copy(reshape.897), metadata={op_name="XLA_Retvals"}
  get-tuple-element.731 = f32[64]{0} get-tuple-element(arg_tuple.723), index=7
  get-tuple-element.724 = f32[64]{0} get-tuple-element(arg_tuple.723), index=0
  broadcast.872 = f32[64]{0} broadcast(get-tuple-element.728), dimensions={}, metadata={op_type="RealDiv" op_name="cond/truediv" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.873 = f32[64]{0} divide(get-tuple-element.724, broadcast.872), metadata={op_type="RealDiv" op_name="cond/truediv" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.884 = f32[64]{0} subtract(divide.873, get-tuple-element.731), metadata={op_type="Sub" op_name="cond/adam/Sub_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.885 = f32[] constant(0.1), metadata={op_type="Mul" op_name="cond/adam/Mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.886 = f32[64]{0} broadcast(constant.885), dimensions={}, metadata={op_type="Mul" op_name="cond/adam/Mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.887 = f32[64]{0} multiply(subtract.884, broadcast.886), metadata={op_type="Mul" op_name="cond/adam/Mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.888 = f32[64]{0} add(get-tuple-element.731, multiply.887), metadata={op_type="AssignAddVariableOp" op_name="cond/adam/AssignAddVariableOp" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.899 = f32[64]{0} reshape(add.888), metadata={op_name="XLA_Retvals"}
  copy.900 = f32[64]{0} copy(reshape.899), metadata={op_name="XLA_Retvals"}
  get-tuple-element.732 = f32[64]{0} get-tuple-element(arg_tuple.723), index=8
  multiply.874 = f32[64]{0} multiply(divide.873, divide.873), metadata={op_type="Square" op_name="cond/adam/Square" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.875 = f32[64]{0} subtract(multiply.874, get-tuple-element.732), metadata={op_type="Sub" op_name="cond/adam/Sub_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.876 = f32[] constant(0.001), metadata={op_type="Mul" op_name="cond/adam/Mul_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.877 = f32[64]{0} broadcast(constant.876), dimensions={}, metadata={op_type="Mul" op_name="cond/adam/Mul_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.878 = f32[64]{0} multiply(subtract.875, broadcast.877), metadata={op_type="Mul" op_name="cond/adam/Mul_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.879 = f32[64]{0} add(get-tuple-element.732, multiply.878), metadata={op_type="AssignAddVariableOp" op_name="cond/adam/AssignAddVariableOp_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.901 = f32[64]{0} reshape(add.879), metadata={op_name="XLA_Retvals"}
  copy.902 = f32[64]{0} copy(reshape.901), metadata={op_name="XLA_Retvals"}
  get-tuple-element.733 = f32[64]{0} get-tuple-element(arg_tuple.723), index=9
  constant.753 = f32[] constant(1), metadata={op_type="Sub" op_name="cond/adam/sub" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.751 = f32[] constant(0.999), metadata={op_type="Pow" op_name="cond/adam/Pow_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.744 = s64[] constant(1), metadata={op_type="AddV2" op_name="cond/adam/add" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.745 = s64[] add(get-tuple-element.730, constant.744), metadata={op_type="AddV2" op_name="cond/adam/add" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.746 = f32[] convert(add.745), metadata={op_type="Cast" op_name="cond/adam/Cast_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  power.752 = f32[] power(constant.751, convert.746), metadata={op_type="Pow" op_name="cond/adam/Pow_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.754 = f32[] subtract(constant.753, power.752), metadata={op_type="Sub" op_name="cond/adam/sub" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  sqrt.755 = f32[] sqrt(subtract.754), metadata={op_type="Sqrt" op_name="cond/adam/Sqrt"}
  multiply.756 = f32[] multiply(get-tuple-element.729, sqrt.755), metadata={op_type="Mul" op_name="cond/adam/mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.749 = f32[] constant(1), metadata={op_type="Sub" op_name="cond/adam/sub_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.747 = f32[] constant(0.9), metadata={op_type="Pow" op_name="cond/adam/Pow" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  power.748 = f32[] power(constant.747, convert.746), metadata={op_type="Pow" op_name="cond/adam/Pow" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.750 = f32[] subtract(constant.749, power.748), metadata={op_type="Sub" op_name="cond/adam/sub_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.757 = f32[] divide(multiply.756, subtract.750), metadata={op_type="RealDiv" op_name="cond/adam/truediv" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.889 = f32[64]{0} broadcast(divide.757), dimensions={}, metadata={op_type="Mul" op_name="cond/adam/Mul_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.890 = f32[64]{0} multiply(add.888, broadcast.889), metadata={op_type="Mul" op_name="cond/adam/Mul_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  sqrt.880 = f32[64]{0} sqrt(add.879), metadata={op_type="Sqrt" op_name="cond/adam/Sqrt_1"}
  constant.881 = f32[] constant(1e-07), metadata={op_type="AddV2" op_name="cond/adam/Add_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.882 = f32[64]{0} broadcast(constant.881), dimensions={}, metadata={op_type="AddV2" op_name="cond/adam/Add_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.883 = f32[64]{0} add(sqrt.880, broadcast.882), metadata={op_type="AddV2" op_name="cond/adam/Add_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.891 = f32[64]{0} divide(multiply.890, add.883), metadata={op_type="RealDiv" op_name="cond/adam/truediv_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.892 = f32[64]{0} subtract(get-tuple-element.733, divide.891), metadata={op_type="AssignSubVariableOp" op_name="cond/adam/AssignSubVariableOp" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.903 = f32[64]{0} reshape(subtract.892), metadata={op_name="XLA_Retvals"}
  copy.904 = f32[64]{0} copy(reshape.903), metadata={op_name="XLA_Retvals"}
  get-tuple-element.734 = f32[64]{0} get-tuple-element(arg_tuple.723), index=10
  get-tuple-element.725 = f32[64]{0} get-tuple-element(arg_tuple.723), index=1
  broadcast.809 = f32[64]{0} broadcast(get-tuple-element.728), dimensions={}, metadata={op_type="RealDiv" op_name="cond/truediv_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.810 = f32[64]{0} divide(get-tuple-element.725, broadcast.809), metadata={op_type="RealDiv" op_name="cond/truediv_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.821 = f32[64]{0} subtract(divide.810, get-tuple-element.734), metadata={op_type="Sub" op_name="cond/adam/Sub_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.822 = f32[] constant(0.1), metadata={op_type="Mul" op_name="cond/adam/Mul_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.823 = f32[64]{0} broadcast(constant.822), dimensions={}, metadata={op_type="Mul" op_name="cond/adam/Mul_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.824 = f32[64]{0} multiply(subtract.821, broadcast.823), metadata={op_type="Mul" op_name="cond/adam/Mul_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.825 = f32[64]{0} add(get-tuple-element.734, multiply.824), metadata={op_type="AssignAddVariableOp" op_name="cond/adam/AssignAddVariableOp_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.905 = f32[64]{0} reshape(add.825), metadata={op_name="XLA_Retvals"}
  copy.906 = f32[64]{0} copy(reshape.905), metadata={op_name="XLA_Retvals"}
  get-tuple-element.735 = f32[64]{0} get-tuple-element(arg_tuple.723), index=11
  multiply.811 = f32[64]{0} multiply(divide.810, divide.810), metadata={op_type="Square" op_name="cond/adam/Square_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.812 = f32[64]{0} subtract(multiply.811, get-tuple-element.735), metadata={op_type="Sub" op_name="cond/adam/Sub_7" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.813 = f32[] constant(0.001), metadata={op_type="Mul" op_name="cond/adam/Mul_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.814 = f32[64]{0} broadcast(constant.813), dimensions={}, metadata={op_type="Mul" op_name="cond/adam/Mul_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.815 = f32[64]{0} multiply(subtract.812, broadcast.814), metadata={op_type="Mul" op_name="cond/adam/Mul_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.816 = f32[64]{0} add(get-tuple-element.735, multiply.815), metadata={op_type="AssignAddVariableOp" op_name="cond/adam/AssignAddVariableOp_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.907 = f32[64]{0} reshape(add.816), metadata={op_name="XLA_Retvals"}
  copy.908 = f32[64]{0} copy(reshape.907), metadata={op_name="XLA_Retvals"}
  get-tuple-element.736 = f32[64]{0} get-tuple-element(arg_tuple.723), index=12
  constant.767 = f32[] constant(1), metadata={op_type="Sub" op_name="cond/adam/sub_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.765 = f32[] constant(0.999), metadata={op_type="Pow" op_name="cond/adam/Pow_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.758 = s64[] constant(1), metadata={op_type="AddV2" op_name="cond/adam/add_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.759 = s64[] add(get-tuple-element.730, constant.758), metadata={op_type="AddV2" op_name="cond/adam/add_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.760 = f32[] convert(add.759), metadata={op_type="Cast" op_name="cond/adam/Cast_9" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  power.766 = f32[] power(constant.765, convert.760), metadata={op_type="Pow" op_name="cond/adam/Pow_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.768 = f32[] subtract(constant.767, power.766), metadata={op_type="Sub" op_name="cond/adam/sub_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  sqrt.769 = f32[] sqrt(subtract.768), metadata={op_type="Sqrt" op_name="cond/adam/Sqrt_2"}
  multiply.770 = f32[] multiply(get-tuple-element.729, sqrt.769), metadata={op_type="Mul" op_name="cond/adam/mul_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.763 = f32[] constant(1), metadata={op_type="Sub" op_name="cond/adam/sub_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.761 = f32[] constant(0.9), metadata={op_type="Pow" op_name="cond/adam/Pow_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  power.762 = f32[] power(constant.761, convert.760), metadata={op_type="Pow" op_name="cond/adam/Pow_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.764 = f32[] subtract(constant.763, power.762), metadata={op_type="Sub" op_name="cond/adam/sub_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.771 = f32[] divide(multiply.770, subtract.764), metadata={op_type="RealDiv" op_name="cond/adam/truediv_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.826 = f32[64]{0} broadcast(divide.771), dimensions={}, metadata={op_type="Mul" op_name="cond/adam/Mul_7" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.827 = f32[64]{0} multiply(add.825, broadcast.826), metadata={op_type="Mul" op_name="cond/adam/Mul_7" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  sqrt.817 = f32[64]{0} sqrt(add.816), metadata={op_type="Sqrt" op_name="cond/adam/Sqrt_3"}
  constant.818 = f32[] constant(1e-07), metadata={op_type="AddV2" op_name="cond/adam/Add_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.819 = f32[64]{0} broadcast(constant.818), dimensions={}, metadata={op_type="AddV2" op_name="cond/adam/Add_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.820 = f32[64]{0} add(sqrt.817, broadcast.819), metadata={op_type="AddV2" op_name="cond/adam/Add_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.828 = f32[64]{0} divide(multiply.827, add.820), metadata={op_type="RealDiv" op_name="cond/adam/truediv_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.829 = f32[64]{0} subtract(get-tuple-element.736, divide.828), metadata={op_type="AssignSubVariableOp" op_name="cond/adam/AssignSubVariableOp_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.909 = f32[64]{0} reshape(subtract.829), metadata={op_name="XLA_Retvals"}
  copy.910 = f32[64]{0} copy(reshape.909), metadata={op_name="XLA_Retvals"}
  get-tuple-element.737 = f32[64]{0} get-tuple-element(arg_tuple.723), index=13
  get-tuple-element.726 = f32[64]{0} get-tuple-element(arg_tuple.723), index=2
  broadcast.830 = f32[64]{0} broadcast(get-tuple-element.728), dimensions={}, metadata={op_type="RealDiv" op_name="cond/truediv_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.831 = f32[64]{0} divide(get-tuple-element.726, broadcast.830), metadata={op_type="RealDiv" op_name="cond/truediv_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.842 = f32[64]{0} subtract(divide.831, get-tuple-element.737), metadata={op_type="Sub" op_name="cond/adam/Sub_10" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.843 = f32[] constant(0.1), metadata={op_type="Mul" op_name="cond/adam/Mul_9" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.844 = f32[64]{0} broadcast(constant.843), dimensions={}, metadata={op_type="Mul" op_name="cond/adam/Mul_9" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.845 = f32[64]{0} multiply(subtract.842, broadcast.844), metadata={op_type="Mul" op_name="cond/adam/Mul_9" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.846 = f32[64]{0} add(get-tuple-element.737, multiply.845), metadata={op_type="AssignAddVariableOp" op_name="cond/adam/AssignAddVariableOp_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.911 = f32[64]{0} reshape(add.846), metadata={op_name="XLA_Retvals"}
  copy.912 = f32[64]{0} copy(reshape.911), metadata={op_name="XLA_Retvals"}
  get-tuple-element.738 = f32[64]{0} get-tuple-element(arg_tuple.723), index=14
  multiply.832 = f32[64]{0} multiply(divide.831, divide.831), metadata={op_type="Square" op_name="cond/adam/Square_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.833 = f32[64]{0} subtract(multiply.832, get-tuple-element.738), metadata={op_type="Sub" op_name="cond/adam/Sub_11" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.834 = f32[] constant(0.001), metadata={op_type="Mul" op_name="cond/adam/Mul_10" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.835 = f32[64]{0} broadcast(constant.834), dimensions={}, metadata={op_type="Mul" op_name="cond/adam/Mul_10" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.836 = f32[64]{0} multiply(subtract.833, broadcast.835), metadata={op_type="Mul" op_name="cond/adam/Mul_10" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.837 = f32[64]{0} add(get-tuple-element.738, multiply.836), metadata={op_type="AssignAddVariableOp" op_name="cond/adam/AssignAddVariableOp_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.913 = f32[64]{0} reshape(add.837), metadata={op_name="XLA_Retvals"}
  copy.914 = f32[64]{0} copy(reshape.913), metadata={op_name="XLA_Retvals"}
  get-tuple-element.739 = f32[64]{0} get-tuple-element(arg_tuple.723), index=15
  constant.781 = f32[] constant(1), metadata={op_type="Sub" op_name="cond/adam/sub_8" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.779 = f32[] constant(0.999), metadata={op_type="Pow" op_name="cond/adam/Pow_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.772 = s64[] constant(1), metadata={op_type="AddV2" op_name="cond/adam/add_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.773 = s64[] add(get-tuple-element.730, constant.772), metadata={op_type="AddV2" op_name="cond/adam/add_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.774 = f32[] convert(add.773), metadata={op_type="Cast" op_name="cond/adam/Cast_17" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  power.780 = f32[] power(constant.779, convert.774), metadata={op_type="Pow" op_name="cond/adam/Pow_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.782 = f32[] subtract(constant.781, power.780), metadata={op_type="Sub" op_name="cond/adam/sub_8" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  sqrt.783 = f32[] sqrt(subtract.782), metadata={op_type="Sqrt" op_name="cond/adam/Sqrt_4"}
  multiply.784 = f32[] multiply(get-tuple-element.729, sqrt.783), metadata={op_type="Mul" op_name="cond/adam/mul_8" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.777 = f32[] constant(1), metadata={op_type="Sub" op_name="cond/adam/sub_9" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.775 = f32[] constant(0.9), metadata={op_type="Pow" op_name="cond/adam/Pow_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  power.776 = f32[] power(constant.775, convert.774), metadata={op_type="Pow" op_name="cond/adam/Pow_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.778 = f32[] subtract(constant.777, power.776), metadata={op_type="Sub" op_name="cond/adam/sub_9" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.785 = f32[] divide(multiply.784, subtract.778), metadata={op_type="RealDiv" op_name="cond/adam/truediv_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.847 = f32[64]{0} broadcast(divide.785), dimensions={}, metadata={op_type="Mul" op_name="cond/adam/Mul_11" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.848 = f32[64]{0} multiply(add.846, broadcast.847), metadata={op_type="Mul" op_name="cond/adam/Mul_11" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  sqrt.838 = f32[64]{0} sqrt(add.837), metadata={op_type="Sqrt" op_name="cond/adam/Sqrt_5"}
  constant.839 = f32[] constant(1e-07), metadata={op_type="AddV2" op_name="cond/adam/Add_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.840 = f32[64]{0} broadcast(constant.839), dimensions={}, metadata={op_type="AddV2" op_name="cond/adam/Add_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.841 = f32[64]{0} add(sqrt.838, broadcast.840), metadata={op_type="AddV2" op_name="cond/adam/Add_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.849 = f32[64]{0} divide(multiply.848, add.841), metadata={op_type="RealDiv" op_name="cond/adam/truediv_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.850 = f32[64]{0} subtract(get-tuple-element.739, divide.849), metadata={op_type="AssignSubVariableOp" op_name="cond/adam/AssignSubVariableOp_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.915 = f32[64]{0} reshape(subtract.850), metadata={op_name="XLA_Retvals"}
  copy.916 = f32[64]{0} copy(reshape.915), metadata={op_name="XLA_Retvals"}
  get-tuple-element.740 = f32[64]{0} get-tuple-element(arg_tuple.723), index=16
  get-tuple-element.727 = f32[64]{0} get-tuple-element(arg_tuple.723), index=3
  broadcast.851 = f32[64]{0} broadcast(get-tuple-element.728), dimensions={}, metadata={op_type="RealDiv" op_name="cond/truediv_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.852 = f32[64]{0} divide(get-tuple-element.727, broadcast.851), metadata={op_type="RealDiv" op_name="cond/truediv_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.863 = f32[64]{0} subtract(divide.852, get-tuple-element.740), metadata={op_type="Sub" op_name="cond/adam/Sub_14" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.864 = f32[] constant(0.1), metadata={op_type="Mul" op_name="cond/adam/Mul_13" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.865 = f32[64]{0} broadcast(constant.864), dimensions={}, metadata={op_type="Mul" op_name="cond/adam/Mul_13" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.866 = f32[64]{0} multiply(subtract.863, broadcast.865), metadata={op_type="Mul" op_name="cond/adam/Mul_13" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.867 = f32[64]{0} add(get-tuple-element.740, multiply.866), metadata={op_type="AssignAddVariableOp" op_name="cond/adam/AssignAddVariableOp_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.917 = f32[64]{0} reshape(add.867), metadata={op_name="XLA_Retvals"}
  copy.918 = f32[64]{0} copy(reshape.917), metadata={op_name="XLA_Retvals"}
  get-tuple-element.741 = f32[64]{0} get-tuple-element(arg_tuple.723), index=17
  multiply.853 = f32[64]{0} multiply(divide.852, divide.852), metadata={op_type="Square" op_name="cond/adam/Square_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.854 = f32[64]{0} subtract(multiply.853, get-tuple-element.741), metadata={op_type="Sub" op_name="cond/adam/Sub_15" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.855 = f32[] constant(0.001), metadata={op_type="Mul" op_name="cond/adam/Mul_14" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.856 = f32[64]{0} broadcast(constant.855), dimensions={}, metadata={op_type="Mul" op_name="cond/adam/Mul_14" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.857 = f32[64]{0} multiply(subtract.854, broadcast.856), metadata={op_type="Mul" op_name="cond/adam/Mul_14" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.858 = f32[64]{0} add(get-tuple-element.741, multiply.857), metadata={op_type="AssignAddVariableOp" op_name="cond/adam/AssignAddVariableOp_7" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.919 = f32[64]{0} reshape(add.858), metadata={op_name="XLA_Retvals"}
  copy.920 = f32[64]{0} copy(reshape.919), metadata={op_name="XLA_Retvals"}
  get-tuple-element.742 = f32[64]{0} get-tuple-element(arg_tuple.723), index=18
  constant.795 = f32[] constant(1), metadata={op_type="Sub" op_name="cond/adam/sub_12" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.793 = f32[] constant(0.999), metadata={op_type="Pow" op_name="cond/adam/Pow_7" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.786 = s64[] constant(1), metadata={op_type="AddV2" op_name="cond/adam/add_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.787 = s64[] add(get-tuple-element.730, constant.786), metadata={op_type="AddV2" op_name="cond/adam/add_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.788 = f32[] convert(add.787), metadata={op_type="Cast" op_name="cond/adam/Cast_25" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  power.794 = f32[] power(constant.793, convert.788), metadata={op_type="Pow" op_name="cond/adam/Pow_7" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.796 = f32[] subtract(constant.795, power.794), metadata={op_type="Sub" op_name="cond/adam/sub_12" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  sqrt.797 = f32[] sqrt(subtract.796), metadata={op_type="Sqrt" op_name="cond/adam/Sqrt_6"}
  multiply.798 = f32[] multiply(get-tuple-element.729, sqrt.797), metadata={op_type="Mul" op_name="cond/adam/mul_12" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.791 = f32[] constant(1), metadata={op_type="Sub" op_name="cond/adam/sub_13" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.789 = f32[] constant(0.9), metadata={op_type="Pow" op_name="cond/adam/Pow_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  power.790 = f32[] power(constant.789, convert.788), metadata={op_type="Pow" op_name="cond/adam/Pow_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.792 = f32[] subtract(constant.791, power.790), metadata={op_type="Sub" op_name="cond/adam/sub_13" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.799 = f32[] divide(multiply.798, subtract.792), metadata={op_type="RealDiv" op_name="cond/adam/truediv_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.868 = f32[64]{0} broadcast(divide.799), dimensions={}, metadata={op_type="Mul" op_name="cond/adam/Mul_15" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.869 = f32[64]{0} multiply(add.867, broadcast.868), metadata={op_type="Mul" op_name="cond/adam/Mul_15" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  sqrt.859 = f32[64]{0} sqrt(add.858), metadata={op_type="Sqrt" op_name="cond/adam/Sqrt_7"}
  constant.860 = f32[] constant(1e-07), metadata={op_type="AddV2" op_name="cond/adam/Add_7" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.861 = f32[64]{0} broadcast(constant.860), dimensions={}, metadata={op_type="AddV2" op_name="cond/adam/Add_7" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.862 = f32[64]{0} add(sqrt.859, broadcast.861), metadata={op_type="AddV2" op_name="cond/adam/Add_7" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.870 = f32[64]{0} divide(multiply.869, add.862), metadata={op_type="RealDiv" op_name="cond/adam/truediv_7" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.871 = f32[64]{0} subtract(get-tuple-element.742, divide.870), metadata={op_type="AssignSubVariableOp" op_name="cond/adam/AssignSubVariableOp_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.921 = f32[64]{0} reshape(subtract.871), metadata={op_name="XLA_Retvals"}
  copy.922 = f32[64]{0} copy(reshape.921), metadata={op_name="XLA_Retvals"}
  get-tuple-element.808 = s64[] get-tuple-element(conditional.805), index=0, metadata={op_type="If" op_name="cond/cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.923 = s64[] reshape(get-tuple-element.808), metadata={op_name="XLA_Retvals"}
  copy.924 = s64[] copy(reshape.923), metadata={op_name="XLA_Retvals"}
  ROOT tuple.925 = (f32[], f32[], s64[], f32[64]{0}, f32[64]{0}, /*index=5*/f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=10*/f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=15*/s64[]) tuple(copy.894, copy.896, copy.898, copy.900, copy.902, /*index=5*/copy.904, copy.906, copy.908, copy.910, copy.912, /*index=10*/copy.914, copy.916, copy.918, copy.920, copy.922, /*index=15*/copy.924), metadata={op_name="XLA_Retvals"}
} // cond_true_1233_rearrange_0__.722

cond_false_1234_rearrange_0__.926 {
  arg_tuple.927 = (f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[], /*index=5*/f32[], s64[], f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=10*/f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=15*/f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, s64[]) parameter(0), metadata={op_name="XLA_Args"}
  get-tuple-element.928 = f32[64]{0} get-tuple-element(arg_tuple.927), index=0
  get-tuple-element.929 = f32[64]{0} get-tuple-element(arg_tuple.927), index=1
  get-tuple-element.930 = f32[64]{0} get-tuple-element(arg_tuple.927), index=2
  get-tuple-element.931 = f32[64]{0} get-tuple-element(arg_tuple.927), index=3
  get-tuple-element.947 = s64[] get-tuple-element(arg_tuple.927), index=19
  get-tuple-element.932 = f32[] get-tuple-element(arg_tuple.927), index=4
  constant.949 = f32[] constant(2), metadata={op_type="RealDiv" op_name="cond/truediv" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.950 = f32[] divide(get-tuple-element.932, constant.949), metadata={op_type="RealDiv" op_name="cond/truediv" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.951 = f32[] reshape(divide.950), metadata={op_name="XLA_Retvals"}
  copy.952 = f32[] copy(reshape.951), metadata={op_name="XLA_Retvals"}
  get-tuple-element.933 = f32[] get-tuple-element(arg_tuple.927), index=5
  reshape.953 = f32[] reshape(get-tuple-element.933), metadata={op_name="XLA_Retvals"}
  copy.954 = f32[] copy(reshape.953), metadata={op_name="XLA_Retvals"}
  get-tuple-element.934 = s64[] get-tuple-element(arg_tuple.927), index=6
  reshape.955 = s64[] reshape(get-tuple-element.934), metadata={op_name="XLA_Retvals"}
  copy.956 = s64[] copy(reshape.955), metadata={op_name="XLA_Retvals"}
  get-tuple-element.935 = f32[64]{0} get-tuple-element(arg_tuple.927), index=7
  reshape.957 = f32[64]{0} reshape(get-tuple-element.935), metadata={op_name="XLA_Retvals"}
  copy.958 = f32[64]{0} copy(reshape.957), metadata={op_name="XLA_Retvals"}
  get-tuple-element.936 = f32[64]{0} get-tuple-element(arg_tuple.927), index=8
  reshape.959 = f32[64]{0} reshape(get-tuple-element.936), metadata={op_name="XLA_Retvals"}
  copy.960 = f32[64]{0} copy(reshape.959), metadata={op_name="XLA_Retvals"}
  get-tuple-element.937 = f32[64]{0} get-tuple-element(arg_tuple.927), index=9
  reshape.961 = f32[64]{0} reshape(get-tuple-element.937), metadata={op_name="XLA_Retvals"}
  copy.962 = f32[64]{0} copy(reshape.961), metadata={op_name="XLA_Retvals"}
  get-tuple-element.938 = f32[64]{0} get-tuple-element(arg_tuple.927), index=10
  reshape.963 = f32[64]{0} reshape(get-tuple-element.938), metadata={op_name="XLA_Retvals"}
  copy.964 = f32[64]{0} copy(reshape.963), metadata={op_name="XLA_Retvals"}
  get-tuple-element.939 = f32[64]{0} get-tuple-element(arg_tuple.927), index=11
  reshape.965 = f32[64]{0} reshape(get-tuple-element.939), metadata={op_name="XLA_Retvals"}
  copy.966 = f32[64]{0} copy(reshape.965), metadata={op_name="XLA_Retvals"}
  get-tuple-element.940 = f32[64]{0} get-tuple-element(arg_tuple.927), index=12
  reshape.967 = f32[64]{0} reshape(get-tuple-element.940), metadata={op_name="XLA_Retvals"}
  copy.968 = f32[64]{0} copy(reshape.967), metadata={op_name="XLA_Retvals"}
  get-tuple-element.941 = f32[64]{0} get-tuple-element(arg_tuple.927), index=13
  reshape.969 = f32[64]{0} reshape(get-tuple-element.941), metadata={op_name="XLA_Retvals"}
  copy.970 = f32[64]{0} copy(reshape.969), metadata={op_name="XLA_Retvals"}
  get-tuple-element.942 = f32[64]{0} get-tuple-element(arg_tuple.927), index=14
  reshape.971 = f32[64]{0} reshape(get-tuple-element.942), metadata={op_name="XLA_Retvals"}
  copy.972 = f32[64]{0} copy(reshape.971), metadata={op_name="XLA_Retvals"}
  get-tuple-element.943 = f32[64]{0} get-tuple-element(arg_tuple.927), index=15
  reshape.973 = f32[64]{0} reshape(get-tuple-element.943), metadata={op_name="XLA_Retvals"}
  copy.974 = f32[64]{0} copy(reshape.973), metadata={op_name="XLA_Retvals"}
  get-tuple-element.944 = f32[64]{0} get-tuple-element(arg_tuple.927), index=16
  reshape.975 = f32[64]{0} reshape(get-tuple-element.944), metadata={op_name="XLA_Retvals"}
  copy.976 = f32[64]{0} copy(reshape.975), metadata={op_name="XLA_Retvals"}
  get-tuple-element.945 = f32[64]{0} get-tuple-element(arg_tuple.927), index=17
  reshape.977 = f32[64]{0} reshape(get-tuple-element.945), metadata={op_name="XLA_Retvals"}
  copy.978 = f32[64]{0} copy(reshape.977), metadata={op_name="XLA_Retvals"}
  get-tuple-element.946 = f32[64]{0} get-tuple-element(arg_tuple.927), index=18
  reshape.979 = f32[64]{0} reshape(get-tuple-element.946), metadata={op_name="XLA_Retvals"}
  copy.980 = f32[64]{0} copy(reshape.979), metadata={op_name="XLA_Retvals"}
  constant.948 = s64[] constant(0), metadata={op_type="AssignVariableOp" op_name="cond/AssignVariableOp" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.981 = s64[] reshape(constant.948), metadata={op_name="XLA_Retvals"}
  copy.982 = s64[] copy(reshape.981), metadata={op_name="XLA_Retvals"}
  ROOT tuple.983 = (f32[], f32[], s64[], f32[64]{0}, f32[64]{0}, /*index=5*/f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=10*/f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=15*/s64[]) tuple(copy.952, copy.954, copy.956, copy.958, copy.960, /*index=5*/copy.962, copy.964, copy.966, copy.968, copy.970, /*index=10*/copy.972, copy.974, copy.976, copy.978, copy.980, /*index=15*/copy.982), metadata={op_name="XLA_Retvals"}
} // cond_false_1234_rearrange_0__.926

ENTRY a_inference_one_step_on_data_1486__.1059 {
  constant.31 = s32[] constant(768), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.32 = s32[] convert(constant.31), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.33 = s32[1]{0} broadcast(convert.32), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.34 = s32[] constant(96), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.35 = s32[] convert(constant.34), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.36 = s32[1]{0} broadcast(convert.35), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.37 = s32[] constant(96), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.38 = s32[] convert(constant.37), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.39 = s32[1]{0} broadcast(convert.38), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.40 = s32[] constant(64), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.41 = s32[] convert(constant.40), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.42 = s32[1]{0} broadcast(convert.41), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  concatenate.43 = s32[4]{0} concatenate(broadcast.33, broadcast.36, broadcast.39, broadcast.42), dimensions={0}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.50 = s32[] constant(768), metadata={op_type="Shape" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.51 = s32[] convert(constant.50), metadata={op_type="Shape" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.52 = s32[1]{0} broadcast(convert.51), dimensions={}, metadata={op_type="Shape" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  concatenate.53 = s32[1]{0} concatenate(broadcast.52), dimensions={0}, metadata={op_type="Shape" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.149 = s32[] constant(768), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.150 = s32[] convert(constant.149), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.151 = s32[1]{0} broadcast(convert.150), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.152 = s32[] constant(96), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.153 = s32[] convert(constant.152), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.154 = s32[1]{0} broadcast(convert.153), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.155 = s32[] constant(96), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.156 = s32[] convert(constant.155), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.157 = s32[1]{0} broadcast(convert.156), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.158 = s32[] constant(64), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.159 = s32[] convert(constant.158), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.160 = s32[1]{0} broadcast(convert.159), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  concatenate.161 = s32[4]{0} concatenate(broadcast.151, broadcast.154, broadcast.157, broadcast.160), dimensions={0}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.176 = s32[] constant(768), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.177 = s32[] convert(constant.176), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.178 = s32[1]{0} broadcast(convert.177), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.179 = s32[] constant(48), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.180 = s32[] convert(constant.179), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.181 = s32[1]{0} broadcast(convert.180), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.182 = s32[] constant(48), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.183 = s32[] convert(constant.182), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.184 = s32[1]{0} broadcast(convert.183), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.185 = s32[] constant(64), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.186 = s32[] convert(constant.185), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.187 = s32[1]{0} broadcast(convert.186), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  concatenate.188 = s32[4]{0} concatenate(broadcast.178, broadcast.181, broadcast.184, broadcast.187), dimensions={0}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.189 = s32[] constant(768), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.190 = s32[] convert(constant.189), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.191 = s32[1]{0} broadcast(convert.190), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.192 = s32[] constant(48), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.193 = s32[] convert(constant.192), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.194 = s32[1]{0} broadcast(convert.193), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.195 = s32[] constant(48), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.196 = s32[] convert(constant.195), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.197 = s32[1]{0} broadcast(convert.196), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.198 = s32[] constant(64), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.199 = s32[] convert(constant.198), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.200 = s32[1]{0} broadcast(convert.199), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  concatenate.201 = s32[4]{0} concatenate(broadcast.191, broadcast.194, broadcast.197, broadcast.200), dimensions={0}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.227 = s32[] constant(768), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.228 = s32[] convert(constant.227), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.229 = s32[1]{0} broadcast(convert.228), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.230 = s32[] constant(48), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.231 = s32[] convert(constant.230), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.232 = s32[1]{0} broadcast(convert.231), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.233 = s32[] constant(48), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.234 = s32[] convert(constant.233), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.235 = s32[1]{0} broadcast(convert.234), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.236 = s32[] constant(64), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.237 = s32[] convert(constant.236), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.238 = s32[1]{0} broadcast(convert.237), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  concatenate.239 = s32[4]{0} concatenate(broadcast.229, broadcast.232, broadcast.235, broadcast.238), dimensions={0}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.269 = s32[] constant(768), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.270 = s32[] convert(constant.269), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.271 = s32[1]{0} broadcast(convert.270), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.272 = s32[] constant(48), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.273 = s32[] convert(constant.272), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.274 = s32[1]{0} broadcast(convert.273), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.275 = s32[] constant(48), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.276 = s32[] convert(constant.275), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.277 = s32[1]{0} broadcast(convert.276), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.278 = s32[] constant(64), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.279 = s32[] convert(constant.278), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.280 = s32[1]{0} broadcast(convert.279), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  concatenate.281 = s32[4]{0} concatenate(broadcast.271, broadcast.274, broadcast.277, broadcast.280), dimensions={0}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.333 = s32[] constant(768), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.334 = s32[] convert(constant.333), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.335 = s32[1]{0} broadcast(convert.334), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.336 = s32[] constant(48), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.337 = s32[] convert(constant.336), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.338 = s32[1]{0} broadcast(convert.337), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.339 = s32[] constant(48), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.340 = s32[] convert(constant.339), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.341 = s32[1]{0} broadcast(convert.340), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.342 = s32[] constant(64), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.343 = s32[] convert(constant.342), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.344 = s32[1]{0} broadcast(convert.343), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  concatenate.345 = s32[4]{0} concatenate(broadcast.335, broadcast.338, broadcast.341, broadcast.344), dimensions={0}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/add_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.354 = s32[] constant(768), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/flatten_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.355 = s32[] convert(constant.354), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/flatten_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.356 = s32[1]{0} broadcast(convert.355), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/flatten_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.357 = s32[] constant(48), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/flatten_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.358 = s32[] convert(constant.357), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/flatten_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.359 = s32[1]{0} broadcast(convert.358), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/flatten_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.360 = s32[] constant(48), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/flatten_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.361 = s32[] convert(constant.360), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/flatten_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.362 = s32[1]{0} broadcast(convert.361), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/flatten_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.363 = s32[] constant(64), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/flatten_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.364 = s32[] convert(constant.363), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/flatten_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.365 = s32[1]{0} broadcast(convert.364), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/flatten_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  concatenate.366 = s32[4]{0} concatenate(broadcast.356, broadcast.359, broadcast.362, broadcast.365), dimensions={0}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/flatten_1/Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.489 = f32[] constant(0), metadata={op_type="AddV2" op_name="compile_loss/add" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.487 = f32[] constant(1), metadata={op_type="Mul" op_name="compile_loss/mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  arg1.2 = s32[768]{0} parameter(1), parameter_replication={false}, metadata={op_name="XLA_Args"}
  reshape.28 = s32[768]{0} reshape(arg1.2)
  convert.48 = f32[768]{0} convert(reshape.28), metadata={op_type="Cast" op_name="compile_loss/sparse_categorical_crossentropy/Cast_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.49 = s64[768]{0} convert(convert.48), metadata={op_type="Cast" op_name="compile_loss/sparse_categorical_crossentropy/Cast_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.415 = s64[768,147456]{1,0} broadcast(convert.49), dimensions={0}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  iota.414 = s64[768,147456]{1,0} iota(), iota_dimension=1, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  compare.416 = pred[768,147456]{1,0} compare(broadcast.415, iota.414), direction=EQ, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.412 = f32[] constant(1), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.417 = f32[768,147456]{1,0} broadcast(constant.412), dimensions={}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.413 = f32[] constant(0), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.418 = f32[768,147456]{1,0} broadcast(constant.413), dimensions={}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  select.419 = f32[768,147456]{1,0} select(compare.416, broadcast.417, broadcast.418), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.420 = s64[] constant(0), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.421 = s64[768]{0} broadcast(constant.420), dimensions={}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  compare.422 = pred[768]{0} compare(broadcast.421, convert.49), direction=LE, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.423 = s64[] constant(147456), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.424 = s64[768]{0} broadcast(constant.423), dimensions={}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  compare.425 = pred[768]{0} compare(convert.49, broadcast.424), direction=LT, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  and.426 = pred[768]{0} and(compare.422, compare.425), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.427 = f32[] constant(0), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.428 = f32[768]{0} broadcast(constant.427), dimensions={}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.429 = f32[] constant(nan), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.430 = f32[768]{0} broadcast(constant.429), dimensions={}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  select.431 = f32[768]{0} select(and.426, broadcast.428, broadcast.430), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.432 = f32[768,147456]{1,0} broadcast(select.431), dimensions={0}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.433 = f32[768,147456]{1,0} add(select.419, broadcast.432), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  negate.460 = f32[768,147456]{1,0} negate(add.433), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.454 = f32[] constant(0), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.455 = f32[768,147456]{1,0} broadcast(constant.454), dimensions={}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  compare.456 = pred[768,147456]{1,0} compare(add.433, broadcast.455), direction=EQ, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.457 = f32[] constant(0), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.458 = f32[768,147456]{1,0} broadcast(constant.457), dimensions={}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  arg0.1 = f32[768,96,96,64]{3,2,1,0} parameter(0), parameter_replication={false}, metadata={op_name="XLA_Args"}
  reshape.27 = f32[768,96,96,64]{3,2,1,0} reshape(arg0.1)
  convert.29 = f16[768,96,96,64]{3,2,1,0} convert(reshape.27), metadata={op_type="Cast" op_name="sequential_1/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.30 = f32[768,96,96,64]{3,2,1,0} convert(convert.29), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.62 = f32[768,96,96,64]{3,2,1,0} convert(convert.30), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.63 = f32[] constant(0), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.64 = f32[] convert(constant.63), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.69 = f32[64]{0} reduce(convert.62, convert.64), dimensions={0,1,2}, to_apply=sequential_1_batch_normalization_1_moments_mean-reduction.65, metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.70 = s32[] constant(768), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.71 = s32[] constant(96), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.72 = s32[] multiply(constant.70, constant.71), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.73 = s32[] constant(96), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.74 = s32[] multiply(multiply.72, constant.73), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.75 = f32[] convert(multiply.74), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.76 = f32[64]{0} broadcast(convert.75), dimensions={}, metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.77 = f32[64]{0} divide(reduce.69, broadcast.76), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.78 = f32[64]{0} convert(divide.77), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.79 = f32[1,1,1,64]{3,2,1,0} reshape(convert.78), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.81 = f32[64]{0} reshape(reshape.79), metadata={op_type="SquaredDifference" op_name="sequential_1/batch_normalization_1/moments/SquaredDifference" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.82 = f32[768,96,96,64]{3,2,1,0} broadcast(reshape.81), dimensions={3}, metadata={op_type="SquaredDifference" op_name="sequential_1/batch_normalization_1/moments/SquaredDifference" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.83 = f32[768,96,96,64]{3,2,1,0} subtract(convert.30, broadcast.82), metadata={op_type="SquaredDifference" op_name="sequential_1/batch_normalization_1/moments/SquaredDifference" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.84 = f32[768,96,96,64]{3,2,1,0} multiply(subtract.83, subtract.83), metadata={op_type="SquaredDifference" op_name="sequential_1/batch_normalization_1/moments/SquaredDifference" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.85 = f32[768,96,96,64]{3,2,1,0} convert(multiply.84), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.86 = f32[] constant(0), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.87 = f32[] convert(constant.86), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.92 = f32[64]{0} reduce(convert.85, convert.87), dimensions={0,1,2}, to_apply=sequential_1_batch_normalization_1_moments_variance-reduction.88, metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.93 = s32[] constant(768), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.94 = s32[] constant(96), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.95 = s32[] multiply(constant.93, constant.94), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.96 = s32[] constant(96), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.97 = s32[] multiply(multiply.95, constant.96), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.98 = f32[] convert(multiply.97), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.99 = f32[64]{0} broadcast(convert.98), dimensions={}, metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.100 = f32[64]{0} divide(reduce.92, broadcast.99), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.101 = f32[64]{0} convert(divide.100), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.102 = f32[1,1,1,64]{3,2,1,0} reshape(convert.101), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.103 = f32[64]{0} reshape(reshape.102), metadata={op_type="Squeeze" op_name="sequential_1/batch_normalization_1/moments/Squeeze_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.104 = f32[] constant(0.001), metadata={op_type="AddV2" op_name="sequential_1/batch_normalization_1/batchnorm/add" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.105 = f32[64]{0} broadcast(constant.104), dimensions={}, metadata={op_type="AddV2" op_name="sequential_1/batch_normalization_1/batchnorm/add" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.106 = f32[64]{0} add(reshape.103, broadcast.105), metadata={op_type="AddV2" op_name="sequential_1/batch_normalization_1/batchnorm/add" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  rsqrt.107 = f32[64]{0} rsqrt(add.106), metadata={op_type="Rsqrt" op_name="sequential_1/batch_normalization_1/batchnorm/Rsqrt"}
  arg4.5 = f32[64]{0} parameter(4), parameter_replication={false}, metadata={op_name="XLA_Args"}
  convert.144 = f16[64]{0} convert(arg4.5), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1/Cast_7/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.145 = f32[64]{0} convert(convert.144), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1/Cast_7" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.146 = f32[64]{0} multiply(rsqrt.107, convert.145), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1/batchnorm/mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.147 = f32[768,96,96,64]{3,2,1,0} broadcast(multiply.146), dimensions={3}, metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1/batchnorm/mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.148 = f32[768,96,96,64]{3,2,1,0} multiply(convert.30, broadcast.147), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1/batchnorm/mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  arg5.6 = f32[64]{0} parameter(5), parameter_replication={false}, metadata={op_name="XLA_Args"}
  convert.163 = f16[64]{0} convert(arg5.6), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1/Cast_8/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.164 = f32[64]{0} convert(convert.163), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1/Cast_8" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.80 = f32[64]{0} reshape(reshape.79), metadata={op_type="Squeeze" op_name="sequential_1/batch_normalization_1/moments/Squeeze" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.162 = f32[64]{0} multiply(reshape.80, multiply.146), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1/batchnorm/mul_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.165 = f32[64]{0} subtract(convert.164, multiply.162), metadata={op_type="Sub" op_name="sequential_1/batch_normalization_1/batchnorm/sub" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.166 = f32[768,96,96,64]{3,2,1,0} broadcast(subtract.165), dimensions={3}, metadata={op_type="AddV2" op_name="sequential_1/batch_normalization_1/batchnorm/add_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.167 = f32[768,96,96,64]{3,2,1,0} add(multiply.148, broadcast.166), metadata={op_type="AddV2" op_name="sequential_1/batch_normalization_1/batchnorm/add_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.168 = f16[768,96,96,64]{3,2,1,0} convert(add.167), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1/Cast_9" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.169 = f16[] constant(-inf), metadata={op_type="MaxPool" op_name="sequential_1/max_pooling2d_1/MaxPool2d" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce-window.174 = f16[768,48,48,64]{3,2,1,0} reduce-window(convert.168, constant.169), window={size=1x3x3x1 stride=1x2x2x1 pad=0_0x0_1x0_1x0_0}, to_apply=max_F16.170, metadata={op_type="MaxPool" op_name="sequential_1/max_pooling2d_1/MaxPool2d" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.175 = f32[768,48,48,64]{3,2,1,0} convert(reduce-window.174), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1_2/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.240 = f32[768,48,48,64]{3,2,1,0} convert(convert.175), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.241 = f32[] constant(0), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.242 = f32[] convert(constant.241), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.247 = f32[64]{0} reduce(convert.240, convert.242), dimensions={0,1,2}, to_apply=sequential_1_batch_normalization_1_2_moments_mean-reduction.243, metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.248 = s32[] constant(768), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.249 = s32[] constant(48), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.250 = s32[] multiply(constant.248, constant.249), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.251 = s32[] constant(48), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.252 = s32[] multiply(multiply.250, constant.251), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.253 = f32[] convert(multiply.252), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.254 = f32[64]{0} broadcast(convert.253), dimensions={}, metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.255 = f32[64]{0} divide(reduce.247, broadcast.254), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.256 = f32[64]{0} convert(divide.255), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.257 = f32[1,1,1,64]{3,2,1,0} reshape(convert.256), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/mean" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.265 = f32[64]{0} reshape(reshape.257), metadata={op_type="SquaredDifference" op_name="sequential_1/batch_normalization_1_2/moments/SquaredDifference" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.266 = f32[768,48,48,64]{3,2,1,0} broadcast(reshape.265), dimensions={3}, metadata={op_type="SquaredDifference" op_name="sequential_1/batch_normalization_1_2/moments/SquaredDifference" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.267 = f32[768,48,48,64]{3,2,1,0} subtract(convert.175, broadcast.266), metadata={op_type="SquaredDifference" op_name="sequential_1/batch_normalization_1_2/moments/SquaredDifference" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.268 = f32[768,48,48,64]{3,2,1,0} multiply(subtract.267, subtract.267), metadata={op_type="SquaredDifference" op_name="sequential_1/batch_normalization_1_2/moments/SquaredDifference" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.307 = f32[768,48,48,64]{3,2,1,0} convert(multiply.268), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.308 = f32[] constant(0), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.309 = f32[] convert(constant.308), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.314 = f32[64]{0} reduce(convert.307, convert.309), dimensions={0,1,2}, to_apply=sequential_1_batch_normalization_1_2_moments_variance-reduction.310, metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.315 = s32[] constant(768), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.316 = s32[] constant(48), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.317 = s32[] multiply(constant.315, constant.316), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.318 = s32[] constant(48), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.319 = s32[] multiply(multiply.317, constant.318), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.320 = f32[] convert(multiply.319), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.321 = f32[64]{0} broadcast(convert.320), dimensions={}, metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.322 = f32[64]{0} divide(reduce.314, broadcast.321), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.323 = f32[64]{0} convert(divide.322), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.324 = f32[1,1,1,64]{3,2,1,0} reshape(convert.323), metadata={op_type="Mean" op_name="sequential_1/batch_normalization_1_2/moments/variance" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.325 = f32[64]{0} reshape(reshape.324), metadata={op_type="Squeeze" op_name="sequential_1/batch_normalization_1_2/moments/Squeeze_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.326 = f32[] constant(0.001), metadata={op_type="AddV2" op_name="sequential_1/batch_normalization_1_2/batchnorm/add" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.327 = f32[64]{0} broadcast(constant.326), dimensions={}, metadata={op_type="AddV2" op_name="sequential_1/batch_normalization_1_2/batchnorm/add" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.328 = f32[64]{0} add(reshape.325, broadcast.327), metadata={op_type="AddV2" op_name="sequential_1/batch_normalization_1_2/batchnorm/add" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  rsqrt.329 = f32[64]{0} rsqrt(add.328), metadata={op_type="Rsqrt" op_name="sequential_1/batch_normalization_1_2/batchnorm/Rsqrt"}
  arg8.9 = f32[64]{0} parameter(8), parameter_replication={false}, metadata={op_name="XLA_Args"}
  convert.124 = f16[64]{0} convert(arg8.9), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1_2/Cast_7/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.125 = f32[64]{0} convert(convert.124), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1_2/Cast_7" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.330 = f32[64]{0} multiply(rsqrt.329, convert.125), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1_2/batchnorm/mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.331 = f32[768,48,48,64]{3,2,1,0} broadcast(multiply.330), dimensions={3}, metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1_2/batchnorm/mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.332 = f32[768,48,48,64]{3,2,1,0} multiply(convert.175, broadcast.331), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1_2/batchnorm/mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  arg9.10 = f32[64]{0} parameter(9), parameter_replication={false}, metadata={op_name="XLA_Args"}
  convert.126 = f16[64]{0} convert(arg9.10), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1_2/Cast_8/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.127 = f32[64]{0} convert(convert.126), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1_2/Cast_8" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.258 = f32[64]{0} reshape(reshape.257), metadata={op_type="Squeeze" op_name="sequential_1/batch_normalization_1_2/moments/Squeeze" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.346 = f32[64]{0} multiply(reshape.258, multiply.330), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1_2/batchnorm/mul_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.347 = f32[64]{0} subtract(convert.127, multiply.346), metadata={op_type="Sub" op_name="sequential_1/batch_normalization_1_2/batchnorm/sub" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.348 = f32[768,48,48,64]{3,2,1,0} broadcast(subtract.347), dimensions={3}, metadata={op_type="AddV2" op_name="sequential_1/batch_normalization_1_2/batchnorm/add_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.349 = f32[768,48,48,64]{3,2,1,0} add(multiply.332, broadcast.348), metadata={op_type="AddV2" op_name="sequential_1/batch_normalization_1_2/batchnorm/add_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.350 = f16[768,48,48,64]{3,2,1,0} convert(add.349), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1_2/Cast_9" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.351 = f16[] constant(0), metadata={op_type="Relu" op_name="sequential_1/activation_1/Relu" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.352 = f16[768,48,48,64]{3,2,1,0} broadcast(constant.351), dimensions={}, metadata={op_type="Relu" op_name="sequential_1/activation_1/Relu"}
  maximum.353 = f16[768,48,48,64]{3,2,1,0} maximum(convert.350, broadcast.352), metadata={op_type="Relu" op_name="sequential_1/activation_1/Relu"}
  reshape.367 = f16[768,147456]{1,0} reshape(maximum.353), metadata={op_type="Reshape" op_name="sequential_1/flatten_1/Reshape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.404 = f32[768,147456]{1,0} convert(reshape.367), metadata={op_type="Cast" op_name="compile_loss/sparse_categorical_crossentropy/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.405 = f32[] constant(0.999999881), metadata={op_type="Minimum" op_name="compile_loss/sparse_categorical_crossentropy/clip_by_value/Minimum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.406 = f32[768,147456]{1,0} broadcast(constant.405), dimensions={}, metadata={op_type="Minimum" op_name="compile_loss/sparse_categorical_crossentropy/clip_by_value/Minimum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  minimum.407 = f32[768,147456]{1,0} minimum(convert.404, broadcast.406), metadata={op_type="Minimum" op_name="compile_loss/sparse_categorical_crossentropy/clip_by_value/Minimum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.408 = f32[] constant(1e-07), metadata={op_type="Maximum" op_name="compile_loss/sparse_categorical_crossentropy/clip_by_value" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.409 = f32[768,147456]{1,0} broadcast(constant.408), dimensions={}, metadata={op_type="Maximum" op_name="compile_loss/sparse_categorical_crossentropy/clip_by_value" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  maximum.410 = f32[768,147456]{1,0} maximum(minimum.407, broadcast.409), metadata={op_type="Maximum" op_name="compile_loss/sparse_categorical_crossentropy/clip_by_value" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  log.411 = f32[768,147456]{1,0} log(maximum.410), metadata={op_type="Log" op_name="compile_loss/sparse_categorical_crossentropy/Log" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.434 = f32[] constant(-inf), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.439 = f32[768]{0} reduce(log.411, constant.434), dimensions={1}, to_apply=max_float_.435, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.440 = f32[768,147456]{1,0} broadcast(reduce.439), dimensions={0}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.441 = f32[768,147456]{1,0} subtract(log.411, broadcast.440), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  exponential.442 = f32[768,147456]{1,0} exponential(subtract.441), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.443 = f32[768,147456]{1,0} convert(exponential.442), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.444 = f32[] constant(0), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.449 = f32[768]{0} reduce(convert.443, constant.444), dimensions={1}, to_apply=add_float_.445, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.450 = f32[768]{0} convert(reduce.449), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  log.451 = f32[768]{0} log(convert.450), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.452 = f32[768,147456]{1,0} broadcast(log.451), dimensions={0}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.453 = f32[768,147456]{1,0} subtract(subtract.441, broadcast.452), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  select.459 = f32[768,147456]{1,0} select(compare.456, broadcast.458, subtract.453), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.461 = f32[768,147456]{1,0} multiply(negate.460, select.459), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.462 = f32[768,147456]{1,0} convert(multiply.461), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.463 = f32[] constant(0), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.468 = f32[768]{0} reduce(convert.462, constant.463), dimensions={1}, to_apply=add_float_.464, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.469 = f32[768]{0} convert(reduce.468), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.476 = f32[768]{0} convert(convert.469), metadata={op_type="Sum" op_name="compile_loss/sparse_categorical_crossentropy/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.477 = f32[] constant(0), metadata={op_type="Sum" op_name="compile_loss/sparse_categorical_crossentropy/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.478 = f32[] convert(constant.477), metadata={op_type="Sum" op_name="compile_loss/sparse_categorical_crossentropy/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.483 = f32[] reduce(convert.476, convert.478), dimensions={0}, to_apply=compile_loss_sparse_categorical_crossentropy_Sum-reduction.479, metadata={op_type="Sum" op_name="compile_loss/sparse_categorical_crossentropy/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.484 = f32[] convert(reduce.483), metadata={op_type="Sum" op_name="compile_loss/sparse_categorical_crossentropy/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.485 = f32[] constant(768), metadata={op_type="RealDiv" op_name="compile_loss/sparse_categorical_crossentropy/truediv" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.486 = f32[] divide(convert.484, constant.485), metadata={op_type="RealDiv" op_name="compile_loss/sparse_categorical_crossentropy/truediv" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.488 = f32[] multiply(constant.487, divide.486), metadata={op_type="Mul" op_name="compile_loss/mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.490 = f32[] add(constant.489, multiply.488), metadata={op_type="AddV2" op_name="compile_loss/add" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  arg12.13 = f32[] parameter(12), parameter_replication={false}, metadata={op_name="XLA_Args"}
  multiply.493 = f32[] multiply(add.490, arg12.13), metadata={op_type="Mul" op_name="mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.54 = f32[] constant(1), metadata={op_type="Mul" op_name="gradient_tape/mul/Mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.55 = f32[] multiply(constant.54, arg12.13), metadata={op_type="Mul" op_name="gradient_tape/mul/Mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.494 = f32[] multiply(multiply.55, divide.486), metadata={op_type="Mul" op_name="gradient_tape/compile_loss/mul/Mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.56 = f32[] constant(1), metadata={op_type="Mul" op_name="gradient_tape/compile_loss/mul/Mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.57 = f32[] multiply(multiply.55, constant.56), metadata={op_type="Mul" op_name="gradient_tape/compile_loss/mul/Mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  negate.495 = f32[] negate(convert.484), metadata={op_type="Neg" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/truediv/Neg" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.496 = f32[] constant(768), metadata={op_type="RealDiv" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/truediv/RealDiv_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.497 = f32[] divide(negate.495, constant.496), metadata={op_type="RealDiv" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/truediv/RealDiv_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.498 = f32[] constant(768), metadata={op_type="RealDiv" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/truediv/RealDiv_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.499 = f32[] divide(divide.497, constant.498), metadata={op_type="RealDiv" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/truediv/RealDiv_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.500 = f32[] multiply(multiply.57, divide.499), metadata={op_type="Mul" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/truediv/mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.601 = f16[] constant(-inf), metadata={op_type="MaxPoolGrad" op_name="gradient_tape/sequential_1/max_pooling2d_1/MaxPool2d/MaxPoolGrad" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce-window.606 = f16[768,48,48,64]{3,2,1,0} reduce-window(convert.168, constant.601), window={size=1x3x3x1 stride=1x2x2x1 pad=0_0x0_1x0_1x0_0}, to_apply=max_F16.602, metadata={op_type="MaxPoolGrad" op_name="gradient_tape/sequential_1/max_pooling2d_1/MaxPool2d/MaxPoolGrad" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.526 = f16[] constant(0), metadata={op_type="ReluGrad" op_name="gradient_tape/sequential_1/activation_1/ReluGrad" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.527 = f16[768,48,48,64]{3,2,1,0} broadcast(constant.526), dimensions={}, metadata={op_type="ReluGrad" op_name="gradient_tape/sequential_1/activation_1/ReluGrad"}
  compare.528 = pred[768,48,48,64]{3,2,1,0} compare(maximum.353, broadcast.527), direction=GT, metadata={op_type="ReluGrad" op_name="gradient_tape/sequential_1/activation_1/ReluGrad"}
  constant.520 = f32[] constant(0.999999881), metadata={op_type="LessEqual" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/clip_by_value/LessEqual" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.521 = f32[768,147456]{1,0} broadcast(constant.520), dimensions={}, metadata={op_type="LessEqual" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/clip_by_value/LessEqual" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  compare.522 = pred[768,147456]{1,0} compare(convert.404, broadcast.521), direction=LE, metadata={op_type="LessEqual" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/clip_by_value/LessEqual" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.514 = f32[] constant(1e-07), metadata={op_type="GreaterEqual" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/clip_by_value/GreaterEqual" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.515 = f32[768,147456]{1,0} broadcast(constant.514), dimensions={}, metadata={op_type="GreaterEqual" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/clip_by_value/GreaterEqual" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  compare.516 = pred[768,147456]{1,0} compare(minimum.407, broadcast.515), direction=GE, metadata={op_type="GreaterEqual" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/clip_by_value/GreaterEqual" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.473 = f32[] constant(768), metadata={op_type="RealDiv" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/truediv/RealDiv" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.474 = f32[] divide(multiply.57, constant.473), metadata={op_type="RealDiv" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/truediv/RealDiv" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.475 = f32[1]{0} reshape(divide.474), metadata={op_type="Reshape" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/Reshape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.501 = f32[1]{0} broadcast(reshape.475), dimensions={0}, metadata={op_type="Tile" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/Tile" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.502 = f32[] reshape(broadcast.501), metadata={op_type="Tile" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/Tile" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.503 = f32[768]{0} broadcast(reshape.502), dimensions={}, metadata={op_type="Tile" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/Tile" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.504 = f32[768,1]{1,0} reshape(broadcast.503), metadata={op_type="ExpandDims" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ExpandDims" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.505 = f32[768]{0} reshape(reshape.504), metadata={op_type="Mul" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.506 = f32[768,147456]{1,0} broadcast(reshape.505), dimensions={0}, metadata={op_type="Mul" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.470 = f32[768,147456]{1,0} broadcast(convert.450), dimensions={0}, metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.471 = f32[768,147456]{1,0} divide(exponential.442, broadcast.470), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.472 = f32[768,147456]{1,0} subtract(divide.471, add.433), metadata={op_type="SparseSoftmaxCrossEntropyWithLogits" op_name="compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.507 = f32[768,147456]{1,0} multiply(broadcast.506, subtract.472), metadata={op_type="Mul" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.508 = f32[] constant(1), metadata={op_type="Reciprocal" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/Reciprocal" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.509 = f32[768,147456]{1,0} broadcast(constant.508), dimensions={}, metadata={op_type="Reciprocal" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/Reciprocal"}
  divide.510 = f32[768,147456]{1,0} divide(broadcast.509, maximum.410), metadata={op_type="Reciprocal" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/Reciprocal"}
  multiply.511 = f32[768,147456]{1,0} multiply(multiply.507, divide.510), metadata={op_type="Mul" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.512 = f32[] constant(0), metadata={op_type="ZerosLike" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/clip_by_value/zeros_like" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.513 = f32[768,147456]{1,0} broadcast(constant.512), dimensions={}, metadata={op_type="ZerosLike" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/clip_by_value/zeros_like" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  select.517 = f32[768,147456]{1,0} select(compare.516, multiply.511, broadcast.513), metadata={op_type="SelectV2" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/clip_by_value/SelectV2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.518 = f32[] constant(0), metadata={op_type="ZerosLike" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/clip_by_value/zeros_like_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.519 = f32[768,147456]{1,0} broadcast(constant.518), dimensions={}, metadata={op_type="ZerosLike" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/clip_by_value/zeros_like_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  select.523 = f32[768,147456]{1,0} select(compare.522, select.517, broadcast.519), metadata={op_type="SelectV2" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/clip_by_value/SelectV2_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.524 = f16[768,147456]{1,0} convert(select.523), metadata={op_type="Cast" op_name="gradient_tape/compile_loss/sparse_categorical_crossentropy/Cast/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.525 = f16[768,48,48,64]{3,2,1,0} reshape(convert.524), metadata={op_type="Reshape" op_name="gradient_tape/sequential_1/flatten_1/Reshape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  select.529 = f16[768,48,48,64]{3,2,1,0} select(compare.528, reshape.525, broadcast.527), metadata={op_type="ReluGrad" op_name="gradient_tape/sequential_1/activation_1/ReluGrad"}
  convert.530 = f32[768,48,48,64]{3,2,1,0} convert(select.529), metadata={op_type="Cast" op_name="gradient_tape/sequential_1/batch_normalization_1_2/Cast_9/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.563 = f32[768,48,48,64]{3,2,1,0} broadcast(multiply.330), dimensions={3}, metadata={op_type="Mul" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.564 = f32[768,48,48,64]{3,2,1,0} multiply(convert.530, broadcast.563), metadata={op_type="Mul" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.531 = f32[768,48,48,64]{3,2,1,0} convert(convert.530), metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/add_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.532 = f32[] constant(0), metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/add_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.533 = f32[] convert(constant.532), metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/add_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.538 = f32[64]{0} reduce(convert.531, convert.533), dimensions={0,1,2}, to_apply=gradient_tape_sequential_1_batch_normalization_1_2_batchnorm_add_1_Sum-reduction.534, metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/add_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.539 = f32[64]{0} convert(reduce.538), metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/add_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.540 = f32[1,1,1,64]{3,2,1,0} reshape(convert.539), metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/add_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.541 = f32[64]{0} reshape(reshape.540), metadata={op_type="Reshape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/add_1/Reshape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  negate.554 = f32[64]{0} negate(reshape.541), metadata={op_type="Neg" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/sub/Neg" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.555 = f32[64]{0} multiply(negate.554, multiply.330), metadata={op_type="Mul" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_2/Mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.556 = f32[1,1,1,64]{3,2,1,0} reshape(multiply.555), metadata={op_type="Reshape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Reshape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.557 = f32[1,1,1,64]{3,2,1,0} broadcast(reshape.556), dimensions={0,1,2,3}, metadata={op_type="BroadcastTo" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/BroadcastTo" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.558 = f32[64]{0} reshape(broadcast.557), metadata={op_type="BroadcastTo" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/BroadcastTo" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.559 = f32[768,48,48,64]{3,2,1,0} broadcast(reshape.558), dimensions={3}, metadata={op_type="BroadcastTo" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/BroadcastTo" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.202 = s32[] constant(768), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.203 = s32[] convert(constant.202), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.204 = s32[1]{0} broadcast(convert.203), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.205 = s32[] constant(48), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.206 = s32[] convert(constant.205), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.207 = s32[1]{0} broadcast(convert.206), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.208 = s32[] constant(48), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.209 = s32[] convert(constant.208), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.210 = s32[1]{0} broadcast(convert.209), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.211 = s32[] constant(64), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.212 = s32[] convert(constant.211), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.213 = s32[1]{0} broadcast(convert.212), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  concatenate.214 = s32[4]{0} concatenate(broadcast.204, broadcast.207, broadcast.210, broadcast.213), dimensions={0}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.215 = s32[3]{0} constant({0, 1, 2}), metadata={op_type="GatherV2" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/GatherV2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  gather.216 = s32[3]{0} gather(concatenate.214, constant.215), offset_dims={}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1}, metadata={op_type="GatherV2" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/GatherV2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.217 = s32[3]{0} convert(gather.216), metadata={op_type="Prod" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Prod" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.218 = s32[] constant(1), metadata={op_type="Prod" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Prod" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.219 = s32[] convert(constant.218), metadata={op_type="Prod" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Prod" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.224 = s32[] reduce(convert.217, convert.219), dimensions={0}, to_apply=gradient_tape_sequential_1_batch_normalization_1_2_moments_Prod-reduction.220, metadata={op_type="Prod" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Prod" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.225 = s32[] convert(reduce.224), metadata={op_type="Prod" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Prod" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.226 = f32[] convert(convert.225), metadata={op_type="Cast" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.560 = f32[768,48,48,64]{3,2,1,0} broadcast(convert.226), dimensions={}, metadata={op_type="RealDiv" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/truediv" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.561 = f32[768,48,48,64]{3,2,1,0} divide(broadcast.559, broadcast.560), metadata={op_type="RealDiv" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/truediv" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.598 = f32[768,48,48,64]{3,2,1,0} add(multiply.564, divide.561), metadata={op_type="AddN" op_name="AddN_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.591 = f32[] constant(2), metadata={op_type="Mul" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.592 = f32[768,48,48,64]{3,2,1,0} broadcast(constant.591), dimensions={}, metadata={op_type="Mul" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.579 = f32[64]{0} multiply(rsqrt.329, rsqrt.329), metadata={op_type="RsqrtGrad" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/RsqrtGrad" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.580 = f32[64]{0} multiply(multiply.579, rsqrt.329), metadata={op_type="RsqrtGrad" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/RsqrtGrad" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.565 = f32[768,48,48,64]{3,2,1,0} multiply(convert.175, convert.530), metadata={op_type="Mul" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.566 = f32[768,48,48,64]{3,2,1,0} convert(multiply.565), metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.567 = f32[] constant(0), metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.568 = f32[] convert(constant.567), metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.573 = f32[64]{0} reduce(convert.566, convert.568), dimensions={0,1,2}, to_apply=gradient_tape_sequential_1_batch_normalization_1_2_batchnorm_mul_1_Sum-reduction.569, metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.574 = f32[64]{0} convert(reduce.573), metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.575 = f32[1,1,1,64]{3,2,1,0} reshape(convert.574), metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.576 = f32[64]{0} reshape(reshape.575), metadata={op_type="Reshape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_1/Reshape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.562 = f32[64]{0} multiply(negate.554, reshape.258), metadata={op_type="Mul" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul_2/Mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.577 = f32[64]{0} add(reshape.576, multiply.562), metadata={op_type="AddN" op_name="AddN" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.578 = f32[64]{0} multiply(add.577, convert.125), metadata={op_type="Mul" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul/Mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.581 = f32[] constant(-2), metadata={op_type="RsqrtGrad" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/RsqrtGrad" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.582 = f32[64]{0} broadcast(constant.581), dimensions={}, metadata={op_type="RsqrtGrad" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/RsqrtGrad" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.583 = f32[64]{0} divide(multiply.578, broadcast.582), metadata={op_type="RsqrtGrad" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/RsqrtGrad" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.584 = f32[64]{0} multiply(multiply.580, divide.583), metadata={op_type="RsqrtGrad" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/RsqrtGrad" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.585 = f32[1,1,1,64]{3,2,1,0} reshape(multiply.584), metadata={op_type="Reshape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Reshape_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.586 = f32[1,1,1,64]{3,2,1,0} broadcast(reshape.585), dimensions={0,1,2,3}, metadata={op_type="BroadcastTo" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/BroadcastTo_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.587 = f32[64]{0} reshape(broadcast.586), metadata={op_type="BroadcastTo" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/BroadcastTo_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.588 = f32[768,48,48,64]{3,2,1,0} broadcast(reshape.587), dimensions={3}, metadata={op_type="BroadcastTo" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/BroadcastTo_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.282 = s32[] constant(768), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.283 = s32[] convert(constant.282), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.284 = s32[1]{0} broadcast(convert.283), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.285 = s32[] constant(48), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.286 = s32[] convert(constant.285), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.287 = s32[1]{0} broadcast(convert.286), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.288 = s32[] constant(48), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.289 = s32[] convert(constant.288), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.290 = s32[1]{0} broadcast(convert.289), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.291 = s32[] constant(64), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.292 = s32[] convert(constant.291), metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.293 = s32[1]{0} broadcast(convert.292), dimensions={}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  concatenate.294 = s32[4]{0} concatenate(broadcast.284, broadcast.287, broadcast.290, broadcast.293), dimensions={0}, metadata={op_type="Shape" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Shape_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.295 = s32[3]{0} constant({0, 1, 2}), metadata={op_type="GatherV2" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/GatherV2_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  gather.296 = s32[3]{0} gather(concatenate.294, constant.295), offset_dims={}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1}, metadata={op_type="GatherV2" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/GatherV2_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.297 = s32[3]{0} convert(gather.296), metadata={op_type="Prod" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Prod_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.298 = s32[] constant(1), metadata={op_type="Prod" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Prod_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.299 = s32[] convert(constant.298), metadata={op_type="Prod" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Prod_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.304 = s32[] reduce(convert.297, convert.299), dimensions={0}, to_apply=gradient_tape_sequential_1_batch_normalization_1_2_moments_Prod_1-reduction.300, metadata={op_type="Prod" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Prod_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.305 = s32[] convert(reduce.304), metadata={op_type="Prod" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Prod_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.306 = f32[] convert(convert.305), metadata={op_type="Cast" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Cast_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.589 = f32[768,48,48,64]{3,2,1,0} broadcast(convert.306), dimensions={}, metadata={op_type="RealDiv" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/truediv_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.590 = f32[768,48,48,64]{3,2,1,0} divide(broadcast.588, broadcast.589), metadata={op_type="RealDiv" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/truediv_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.593 = f32[768,48,48,64]{3,2,1,0} multiply(broadcast.592, divide.590), metadata={op_type="Mul" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/Mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.594 = f32[64]{0} reshape(reshape.257), metadata={op_type="Sub" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/sub" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.595 = f32[768,48,48,64]{3,2,1,0} broadcast(reshape.594), dimensions={3}, metadata={op_type="Sub" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/sub" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  subtract.596 = f32[768,48,48,64]{3,2,1,0} subtract(convert.175, broadcast.595), metadata={op_type="Sub" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/sub" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.597 = f32[768,48,48,64]{3,2,1,0} multiply(multiply.593, subtract.596), metadata={op_type="Mul" op_name="gradient_tape/sequential_1/batch_normalization_1_2/moments/mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.599 = f32[768,48,48,64]{3,2,1,0} add(add.598, multiply.597), metadata={op_type="AddN" op_name="AddN_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.600 = f16[768,48,48,64]{3,2,1,0} convert(add.599), metadata={op_type="Cast" op_name="gradient_tape/sequential_1/batch_normalization_1_2/Cast/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.607 = f16[] constant(0), metadata={op_type="MaxPoolGrad" op_name="gradient_tape/sequential_1/max_pooling2d_1/MaxPool2d/MaxPoolGrad" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  select-and-scatter.616 = f16[768,96,96,64]{3,2,1,0} select-and-scatter(convert.168, convert.600, constant.607), window={size=1x3x3x1 stride=1x2x2x1 pad=0_0x0_1x0_1x0_0}, select=ge_F16.608, scatter=add_F16.612, metadata={op_type="MaxPoolGrad" op_name="gradient_tape/sequential_1/max_pooling2d_1/MaxPool2d/MaxPoolGrad" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.617 = f32[768,96,96,64]{3,2,1,0} convert(select-and-scatter.616), metadata={op_type="Cast" op_name="gradient_tape/sequential_1/batch_normalization_1/Cast_9/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.618 = f32[768,96,96,64]{3,2,1,0} convert(convert.617), metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/add_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.619 = f32[] constant(0), metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/add_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.620 = f32[] convert(constant.619), metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/add_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.625 = f32[64]{0} reduce(convert.618, convert.620), dimensions={0,1,2}, to_apply=gradient_tape_sequential_1_batch_normalization_1_batchnorm_add_1_Sum-reduction.621, metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/add_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.626 = f32[64]{0} convert(reduce.625), metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/add_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.627 = f32[1,1,1,64]{3,2,1,0} reshape(convert.626), metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/add_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.628 = f32[64]{0} reshape(reshape.627), metadata={op_type="Reshape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/add_1/Reshape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  negate.641 = f32[64]{0} negate(reshape.628), metadata={op_type="Neg" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/sub/Neg" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.642 = f32[64]{0} multiply(negate.641, multiply.146), metadata={op_type="Mul" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_2/Mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.644 = f32[768,96,96,64]{3,2,1,0} multiply(convert.30, convert.617), metadata={op_type="Mul" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.645 = f32[768,96,96,64]{3,2,1,0} convert(multiply.644), metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.646 = f32[] constant(0), metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.647 = f32[] convert(constant.646), metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.652 = f32[64]{0} reduce(convert.645, convert.647), dimensions={0,1,2}, to_apply=gradient_tape_sequential_1_batch_normalization_1_batchnorm_mul_1_Sum-reduction.648, metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.653 = f32[64]{0} convert(reduce.652), metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.654 = f32[1,1,1,64]{3,2,1,0} reshape(convert.653), metadata={op_type="Sum" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Sum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.655 = f32[64]{0} reshape(reshape.654), metadata={op_type="Reshape" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_1/Reshape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.643 = f32[64]{0} multiply(negate.641, reshape.80), metadata={op_type="Mul" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul_2/Mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.656 = f32[64]{0} add(reshape.655, multiply.643), metadata={op_type="AddN" op_name="AddN_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.657 = f32[64]{0} multiply(add.656, convert.145), metadata={op_type="Mul" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul/Mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.658 = f32[64]{0} multiply(add.656, rsqrt.107), metadata={op_type="Mul" op_name="gradient_tape/sequential_1/batch_normalization_1/batchnorm/mul/Mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.659 = f16[64]{0} convert(multiply.658), metadata={op_type="Cast" op_name="gradient_tape/sequential_1/batch_normalization_1/Cast_7/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.660 = f32[64]{0} convert(convert.659), metadata={op_type="Cast" op_name="gradient_tape/sequential_1/batch_normalization_1/Cast_7/Cast/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  is-finite.661 = pred[64]{0} is-finite(convert.660), metadata={op_type="IsFinite" op_name="IsFinite"}
  convert.662 = pred[64]{0} convert(is-finite.661), metadata={op_type="All" op_name="All" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.663 = pred[] constant(true), metadata={op_type="All" op_name="All" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.664 = pred[] convert(constant.663), metadata={op_type="All" op_name="All" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.669 = pred[] reduce(convert.662, convert.664), dimensions={0}, to_apply=All-reduction.665, metadata={op_type="All" op_name="All" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.670 = pred[] convert(reduce.669), metadata={op_type="All" op_name="All" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.684 = pred[1]{0} reshape(convert.670), metadata={op_type="Pack" op_name="packed" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.629 = f16[64]{0} convert(reshape.628), metadata={op_type="Cast" op_name="gradient_tape/sequential_1/batch_normalization_1/Cast_8/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.630 = f32[64]{0} convert(convert.629), metadata={op_type="Cast" op_name="gradient_tape/sequential_1/batch_normalization_1/Cast_8/Cast/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  is-finite.631 = pred[64]{0} is-finite(convert.630), metadata={op_type="IsFinite" op_name="IsFinite_1"}
  convert.632 = pred[64]{0} convert(is-finite.631), metadata={op_type="All" op_name="All_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.633 = pred[] constant(true), metadata={op_type="All" op_name="All_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.634 = pred[] convert(constant.633), metadata={op_type="All" op_name="All_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.639 = pred[] reduce(convert.632, convert.634), dimensions={0}, to_apply=All_1-reduction.635, metadata={op_type="All" op_name="All_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.640 = pred[] convert(reduce.639), metadata={op_type="All" op_name="All_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.685 = pred[1]{0} reshape(convert.640), metadata={op_type="Pack" op_name="packed" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.671 = f32[64]{0} multiply(add.577, rsqrt.329), metadata={op_type="Mul" op_name="gradient_tape/sequential_1/batch_normalization_1_2/batchnorm/mul/Mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.672 = f16[64]{0} convert(multiply.671), metadata={op_type="Cast" op_name="gradient_tape/sequential_1/batch_normalization_1_2/Cast_7/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.673 = f32[64]{0} convert(convert.672), metadata={op_type="Cast" op_name="gradient_tape/sequential_1/batch_normalization_1_2/Cast_7/Cast/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  is-finite.674 = pred[64]{0} is-finite(convert.673), metadata={op_type="IsFinite" op_name="IsFinite_2"}
  convert.675 = pred[64]{0} convert(is-finite.674), metadata={op_type="All" op_name="All_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.676 = pred[] constant(true), metadata={op_type="All" op_name="All_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.677 = pred[] convert(constant.676), metadata={op_type="All" op_name="All_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.682 = pred[] reduce(convert.675, convert.677), dimensions={0}, to_apply=All_2-reduction.678, metadata={op_type="All" op_name="All_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.683 = pred[] convert(reduce.682), metadata={op_type="All" op_name="All_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.686 = pred[1]{0} reshape(convert.683), metadata={op_type="Pack" op_name="packed" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.542 = f16[64]{0} convert(reshape.541), metadata={op_type="Cast" op_name="gradient_tape/sequential_1/batch_normalization_1_2/Cast_8/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.543 = f32[64]{0} convert(convert.542), metadata={op_type="Cast" op_name="gradient_tape/sequential_1/batch_normalization_1_2/Cast_8/Cast/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  is-finite.544 = pred[64]{0} is-finite(convert.543), metadata={op_type="IsFinite" op_name="IsFinite_3"}
  convert.545 = pred[64]{0} convert(is-finite.544), metadata={op_type="All" op_name="All_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.546 = pred[] constant(true), metadata={op_type="All" op_name="All_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.547 = pred[] convert(constant.546), metadata={op_type="All" op_name="All_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.552 = pred[] reduce(convert.545, convert.547), dimensions={0}, to_apply=All_3-reduction.548, metadata={op_type="All" op_name="All_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.553 = pred[] convert(reduce.552), metadata={op_type="All" op_name="All_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.687 = pred[1]{0} reshape(convert.553), metadata={op_type="Pack" op_name="packed" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  concatenate.688 = pred[4]{0} concatenate(reshape.684, reshape.685, reshape.686, reshape.687), dimensions={0}, metadata={op_type="Pack" op_name="packed" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.689 = pred[4]{0} convert(concatenate.688), metadata={op_type="All" op_name="All_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.690 = pred[] constant(true), metadata={op_type="All" op_name="All_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.691 = pred[] convert(constant.690), metadata={op_type="All" op_name="All_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.696 = pred[] reduce(convert.689, convert.691), dimensions={0}, to_apply=All_4-reduction.692, metadata={op_type="All" op_name="All_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.697 = pred[] convert(reduce.696), metadata={op_type="All" op_name="All_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  arg13.14 = f32[] parameter(13), parameter_replication={false}, metadata={op_name="XLA_Args"}
  arg14.15 = s64[] parameter(14), parameter_replication={false}, metadata={op_name="XLA_Args"}
  arg15.16 = f32[64]{0} parameter(15), parameter_replication={false}, metadata={op_name="XLA_Args"}
  arg16.17 = f32[64]{0} parameter(16), parameter_replication={false}, metadata={op_name="XLA_Args"}
  arg17.18 = f32[64]{0} parameter(17), parameter_replication={false}, metadata={op_name="XLA_Args"}
  arg18.19 = f32[64]{0} parameter(18), parameter_replication={false}, metadata={op_name="XLA_Args"}
  arg19.20 = f32[64]{0} parameter(19), parameter_replication={false}, metadata={op_name="XLA_Args"}
  arg20.21 = f32[64]{0} parameter(20), parameter_replication={false}, metadata={op_name="XLA_Args"}
  arg21.22 = f32[64]{0} parameter(21), parameter_replication={false}, metadata={op_name="XLA_Args"}
  arg22.23 = f32[64]{0} parameter(22), parameter_replication={false}, metadata={op_name="XLA_Args"}
  arg23.24 = s64[] parameter(23), parameter_replication={false}, metadata={op_name="XLA_Args"}
  tuple.698 = (f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[], /*index=5*/f32[], s64[], f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=10*/f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=15*/f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, s64[]) tuple(convert.660, convert.630, convert.673, convert.543, arg12.13, /*index=5*/arg13.14, arg14.15, arg15.16, arg16.17, arg4.5, /*index=10*/arg17.18, arg18.19, arg5.6, arg19.20, arg20.21, /*index=15*/arg8.9, arg21.22, arg22.23, arg9.10, arg23.24), metadata={op_type="If" op_name="cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  conditional.984 = (f32[], f32[], s64[], f32[64]{0}, f32[64]{0}, /*index=5*/f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=10*/f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=15*/s64[]) conditional(convert.697, tuple.698, tuple.698), true_computation=cond_true_1233_rearrange_0__.722, false_computation=cond_false_1234_rearrange_0__.926, metadata={op_type="If" op_name="cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  get-tuple-element.985 = f32[] get-tuple-element(conditional.984), index=0, metadata={op_type="If" op_name="cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  get-tuple-element.999 = s64[] get-tuple-element(conditional.984), index=15, metadata={op_type="If" op_name="cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.44 = s32[] constant(768), metadata={op_type="Shape" op_name="Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.45 = s32[] convert(constant.44), metadata={op_type="Shape" op_name="Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.46 = s32[1]{0} broadcast(convert.45), dimensions={}, metadata={op_type="Shape" op_name="Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  concatenate.47 = s32[1]{0} concatenate(broadcast.46), dimensions={0}, metadata={op_type="Shape" op_name="Shape" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  slice.1008 = s32[1]{0} slice(concatenate.47), slice={[0:1]}, metadata={op_type="StridedSlice" op_name="strided_slice" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1009 = s32[] reshape(slice.1008), metadata={op_type="StridedSlice" op_name="strided_slice" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  arg24.25 = f32[] parameter(24), parameter_replication={false}, metadata={op_name="XLA_Args"}
  iota.370 = s32[768,147456]{1,0} iota(), iota_dimension=1, metadata={op_type="ArgMax" op_name="ArgMax" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.368 = f16[] constant(-inf), metadata={op_type="ArgMax" op_name="ArgMax" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.369 = s32[] constant(0), metadata={op_type="ArgMax" op_name="ArgMax" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.383 = (f16[768]{0}, s32[768]{0}) reduce(reshape.367, iota.370, constant.368, constant.369), dimensions={1}, to_apply=minmax_func.371, metadata={op_type="ArgMax" op_name="ArgMax" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  get-tuple-element.384 = s32[768]{0} get-tuple-element(reduce.383), index=1, metadata={op_type="ArgMax" op_name="ArgMax" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.385 = s64[768]{0} convert(get-tuple-element.384), metadata={op_type="ArgMax" op_name="ArgMax" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.386 = s32[768]{0} convert(convert.385), metadata={op_type="Cast" op_name="Cast_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  compare.387 = pred[768]{0} compare(reshape.28, convert.386), direction=EQ, metadata={op_type="Equal" op_name="Equal" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.388 = f32[768]{0} convert(compare.387), metadata={op_type="Cast" op_name="Cast_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.393 = f32[768]{0} convert(convert.388), metadata={op_type="Sum" op_name="Sum_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.394 = f32[] constant(0), metadata={op_type="Sum" op_name="Sum_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.395 = f32[] convert(constant.394), metadata={op_type="Sum" op_name="Sum_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reduce.400 = f32[] reduce(convert.393, convert.395), dimensions={0}, to_apply=Sum_1-reduction.396, metadata={op_type="Sum" op_name="Sum_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.401 = f32[] convert(reduce.400), metadata={op_type="Sum" op_name="Sum_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.402 = f32[] add(arg24.25, convert.401), metadata={op_type="AddV2" op_name="add_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  arg25.26 = f32[] parameter(25), parameter_replication={false}, metadata={op_name="XLA_Args"}
  constant.389 = f32[] constant(768), metadata={op_type="AddV2" op_name="add_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.390 = f32[] add(arg25.26, constant.389), metadata={op_type="AddV2" op_name="add_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.391 = f32[] constant(1e-07), metadata={op_type="Maximum" op_name="Maximum_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  maximum.392 = f32[] maximum(add.390, constant.391), metadata={op_type="Maximum" op_name="Maximum_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.403 = f32[] divide(add.402, maximum.392), metadata={op_type="RealDiv" op_name="truediv_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1010 = f32[] reshape(divide.403), metadata={op_name="XLA_Retvals"}
  arg10.11 = f32[] parameter(10), parameter_replication={false}, metadata={op_name="XLA_Args"}
  add.491 = f32[] add(arg10.11, add.490), metadata={op_type="AddV2" op_name="add" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  arg11.12 = f32[] parameter(11), parameter_replication={false}, metadata={op_name="XLA_Args"}
  constant.58 = f32[] constant(1), metadata={op_type="AddV2" op_name="add_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.59 = f32[] add(arg11.12, constant.58), metadata={op_type="AddV2" op_name="add_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.60 = f32[] constant(1e-07), metadata={op_type="Maximum" op_name="Maximum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  maximum.61 = f32[] maximum(add.59, constant.60), metadata={op_type="Maximum" op_name="Maximum" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  divide.492 = f32[] divide(add.491, maximum.61), metadata={op_type="RealDiv" op_name="truediv" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1011 = f32[] reshape(divide.492), metadata={op_name="XLA_Retvals"}
  arg2.3 = f32[64]{0} parameter(2), parameter_replication={false}, metadata={op_name="XLA_Args"}
  convert.128 = f16[64]{0} convert(arg2.3), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1/Cast_1/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.129 = f32[64]{0} convert(convert.128), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1/Cast_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.130 = f32[] constant(0.99), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1/mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.131 = f32[64]{0} broadcast(constant.130), dimensions={}, metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1/mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.132 = f32[64]{0} multiply(convert.129, broadcast.131), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1/mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.108 = f32[] constant(0.01), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1/mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.109 = f32[64]{0} broadcast(constant.108), dimensions={}, metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1/mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.110 = f32[64]{0} multiply(reshape.80, broadcast.109), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1/mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.133 = f32[64]{0} add(multiply.132, multiply.110), metadata={op_type="AddV2" op_name="sequential_1/batch_normalization_1/add" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.134 = f16[64]{0} convert(add.133), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1/Cast_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.135 = f32[64]{0} convert(convert.134), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1/Cast_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1012 = f32[64]{0} reshape(convert.135), metadata={op_name="XLA_Retvals"}
  copy.1013 = f32[64]{0} copy(reshape.1012), metadata={op_name="XLA_Retvals"}
  arg3.4 = f32[64]{0} parameter(3), parameter_replication={false}, metadata={op_name="XLA_Args"}
  convert.136 = f16[64]{0} convert(arg3.4), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1/Cast_2/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.137 = f32[64]{0} convert(convert.136), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1/Cast_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.138 = f32[] constant(0.99), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1/mul_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.139 = f32[64]{0} broadcast(constant.138), dimensions={}, metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1/mul_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.140 = f32[64]{0} multiply(convert.137, broadcast.139), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1/mul_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.111 = f32[] constant(0.01), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1/mul_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.112 = f32[64]{0} broadcast(constant.111), dimensions={}, metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1/mul_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.113 = f32[64]{0} multiply(reshape.103, broadcast.112), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1/mul_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.141 = f32[64]{0} add(multiply.140, multiply.113), metadata={op_type="AddV2" op_name="sequential_1/batch_normalization_1/add_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.142 = f16[64]{0} convert(add.141), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1/Cast_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.143 = f32[64]{0} convert(convert.142), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1/Cast_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1014 = f32[64]{0} reshape(convert.143), metadata={op_name="XLA_Retvals"}
  copy.1015 = f32[64]{0} copy(reshape.1014), metadata={op_name="XLA_Retvals"}
  get-tuple-element.989 = f32[64]{0} get-tuple-element(conditional.984), index=5, metadata={op_type="If" op_name="cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1016 = f32[64]{0} reshape(get-tuple-element.989), metadata={op_name="XLA_Retvals"}
  copy.1017 = f32[64]{0} copy(reshape.1016), metadata={op_name="XLA_Retvals"}
  get-tuple-element.992 = f32[64]{0} get-tuple-element(conditional.984), index=8, metadata={op_type="If" op_name="cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1018 = f32[64]{0} reshape(get-tuple-element.992), metadata={op_name="XLA_Retvals"}
  copy.1019 = f32[64]{0} copy(reshape.1018), metadata={op_name="XLA_Retvals"}
  arg6.7 = f32[64]{0} parameter(6), parameter_replication={false}, metadata={op_name="XLA_Args"}
  convert.114 = f16[64]{0} convert(arg6.7), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1_2/Cast_1/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.115 = f32[64]{0} convert(convert.114), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1_2/Cast_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.116 = f32[] constant(0.99), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1_2/mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.117 = f32[64]{0} broadcast(constant.116), dimensions={}, metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1_2/mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.118 = f32[64]{0} multiply(convert.115, broadcast.117), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1_2/mul" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.259 = f32[] constant(0.01), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1_2/mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.260 = f32[64]{0} broadcast(constant.259), dimensions={}, metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1_2/mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.261 = f32[64]{0} multiply(reshape.258, broadcast.260), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1_2/mul_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.262 = f32[64]{0} add(multiply.118, multiply.261), metadata={op_type="AddV2" op_name="sequential_1/batch_normalization_1_2/add" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.263 = f16[64]{0} convert(add.262), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1_2/Cast_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.264 = f32[64]{0} convert(convert.263), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1_2/Cast_4" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1020 = f32[64]{0} reshape(convert.264), metadata={op_name="XLA_Retvals"}
  copy.1021 = f32[64]{0} copy(reshape.1020), metadata={op_name="XLA_Retvals"}
  arg7.8 = f32[64]{0} parameter(7), parameter_replication={false}, metadata={op_name="XLA_Args"}
  convert.119 = f16[64]{0} convert(arg7.8), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1_2/Cast_2/Cast" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.120 = f32[64]{0} convert(convert.119), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1_2/Cast_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.121 = f32[] constant(0.99), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1_2/mul_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.122 = f32[64]{0} broadcast(constant.121), dimensions={}, metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1_2/mul_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.123 = f32[64]{0} multiply(convert.120, broadcast.122), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1_2/mul_2" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  constant.1002 = f32[] constant(0.01), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1_2/mul_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  broadcast.1003 = f32[64]{0} broadcast(constant.1002), dimensions={}, metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1_2/mul_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  multiply.1004 = f32[64]{0} multiply(reshape.325, broadcast.1003), metadata={op_type="Mul" op_name="sequential_1/batch_normalization_1_2/mul_3" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  add.1005 = f32[64]{0} add(multiply.123, multiply.1004), metadata={op_type="AddV2" op_name="sequential_1/batch_normalization_1_2/add_1" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.1006 = f16[64]{0} convert(add.1005), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1_2/Cast_5" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  convert.1007 = f32[64]{0} convert(convert.1006), metadata={op_type="Cast" op_name="sequential_1/batch_normalization_1_2/Cast_6" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1022 = f32[64]{0} reshape(convert.1007), metadata={op_name="XLA_Retvals"}
  copy.1023 = f32[64]{0} copy(reshape.1022), metadata={op_name="XLA_Retvals"}
  get-tuple-element.995 = f32[64]{0} get-tuple-element(conditional.984), index=11, metadata={op_type="If" op_name="cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1024 = f32[64]{0} reshape(get-tuple-element.995), metadata={op_name="XLA_Retvals"}
  copy.1025 = f32[64]{0} copy(reshape.1024), metadata={op_name="XLA_Retvals"}
  get-tuple-element.998 = f32[64]{0} get-tuple-element(conditional.984), index=14, metadata={op_type="If" op_name="cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1026 = f32[64]{0} reshape(get-tuple-element.998), metadata={op_name="XLA_Retvals"}
  copy.1027 = f32[64]{0} copy(reshape.1026), metadata={op_name="XLA_Retvals"}
  reshape.1028 = f32[] reshape(add.491), metadata={op_name="XLA_Retvals"}
  copy.1029 = f32[] copy(reshape.1028), metadata={op_name="XLA_Retvals"}
  reshape.1030 = f32[] reshape(add.59), metadata={op_name="XLA_Retvals"}
  copy.1031 = f32[] copy(reshape.1030), metadata={op_name="XLA_Retvals"}
  get-tuple-element.1000 = f32[] get-tuple-element(conditional.984), index=0, metadata={op_type="If" op_name="cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1032 = f32[] reshape(get-tuple-element.1000), metadata={op_name="XLA_Retvals"}
  copy.1033 = f32[] copy(reshape.1032), metadata={op_name="XLA_Retvals"}
  get-tuple-element.986 = s64[] get-tuple-element(conditional.984), index=2, metadata={op_type="If" op_name="cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1034 = s64[] reshape(get-tuple-element.986), metadata={op_name="XLA_Retvals"}
  copy.1035 = s64[] copy(reshape.1034), metadata={op_name="XLA_Retvals"}
  get-tuple-element.987 = f32[64]{0} get-tuple-element(conditional.984), index=3, metadata={op_type="If" op_name="cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1036 = f32[64]{0} reshape(get-tuple-element.987), metadata={op_name="XLA_Retvals"}
  copy.1037 = f32[64]{0} copy(reshape.1036), metadata={op_name="XLA_Retvals"}
  get-tuple-element.988 = f32[64]{0} get-tuple-element(conditional.984), index=4, metadata={op_type="If" op_name="cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1038 = f32[64]{0} reshape(get-tuple-element.988), metadata={op_name="XLA_Retvals"}
  copy.1039 = f32[64]{0} copy(reshape.1038), metadata={op_name="XLA_Retvals"}
  get-tuple-element.990 = f32[64]{0} get-tuple-element(conditional.984), index=6, metadata={op_type="If" op_name="cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1040 = f32[64]{0} reshape(get-tuple-element.990), metadata={op_name="XLA_Retvals"}
  copy.1041 = f32[64]{0} copy(reshape.1040), metadata={op_name="XLA_Retvals"}
  get-tuple-element.991 = f32[64]{0} get-tuple-element(conditional.984), index=7, metadata={op_type="If" op_name="cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1042 = f32[64]{0} reshape(get-tuple-element.991), metadata={op_name="XLA_Retvals"}
  copy.1043 = f32[64]{0} copy(reshape.1042), metadata={op_name="XLA_Retvals"}
  get-tuple-element.993 = f32[64]{0} get-tuple-element(conditional.984), index=9, metadata={op_type="If" op_name="cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1044 = f32[64]{0} reshape(get-tuple-element.993), metadata={op_name="XLA_Retvals"}
  copy.1045 = f32[64]{0} copy(reshape.1044), metadata={op_name="XLA_Retvals"}
  get-tuple-element.994 = f32[64]{0} get-tuple-element(conditional.984), index=10, metadata={op_type="If" op_name="cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1046 = f32[64]{0} reshape(get-tuple-element.994), metadata={op_name="XLA_Retvals"}
  copy.1047 = f32[64]{0} copy(reshape.1046), metadata={op_name="XLA_Retvals"}
  get-tuple-element.996 = f32[64]{0} get-tuple-element(conditional.984), index=12, metadata={op_type="If" op_name="cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1048 = f32[64]{0} reshape(get-tuple-element.996), metadata={op_name="XLA_Retvals"}
  copy.1049 = f32[64]{0} copy(reshape.1048), metadata={op_name="XLA_Retvals"}
  get-tuple-element.997 = f32[64]{0} get-tuple-element(conditional.984), index=13, metadata={op_type="If" op_name="cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1050 = f32[64]{0} reshape(get-tuple-element.997), metadata={op_name="XLA_Retvals"}
  copy.1051 = f32[64]{0} copy(reshape.1050), metadata={op_name="XLA_Retvals"}
  get-tuple-element.1001 = s64[] get-tuple-element(conditional.984), index=15, metadata={op_type="If" op_name="cond" source_file="/usr/local/lib/python3.9/dist-packages/tensorflow/python/framework/ops.py" source_line=1163}
  reshape.1052 = s64[] reshape(get-tuple-element.1001), metadata={op_name="XLA_Retvals"}
  copy.1053 = s64[] copy(reshape.1052), metadata={op_name="XLA_Retvals"}
  reshape.1054 = f32[] reshape(add.402), metadata={op_name="XLA_Retvals"}
  copy.1055 = f32[] copy(reshape.1054), metadata={op_name="XLA_Retvals"}
  reshape.1056 = f32[] reshape(add.390), metadata={op_name="XLA_Retvals"}
  copy.1057 = f32[] copy(reshape.1056), metadata={op_name="XLA_Retvals"}
  ROOT tuple.1058 = (f32[], f32[], f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=5*/f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=10*/f32[], f32[], f32[], s64[], f32[64]{0}, /*index=15*/f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, f32[64]{0}, /*index=20*/f32[64]{0}, f32[64]{0}, s64[], f32[], f32[]) tuple(reshape.1010, reshape.1011, copy.1013, copy.1015, copy.1017, /*index=5*/copy.1019, copy.1021, copy.1023, copy.1025, copy.1027, /*index=10*/copy.1029, copy.1031, copy.1033, copy.1035, copy.1037, /*index=15*/copy.1039, copy.1041, copy.1043, copy.1045, copy.1047, /*index=20*/copy.1049, copy.1051, copy.1053, copy.1055, copy.1057), metadata={op_name="XLA_Retvals"}
} // a_inference_one_step_on_data_1486__.1059

