??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Exp
x"T
y"T"
Ttype:

2
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
.
Log1p
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
7
Square
x"T
y"T"
Ttype:
2	
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8??
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
Adam/decoder_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/decoder_output/bias/v
?
.Adam/decoder_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/decoder_output/bias/v*
_output_shapes
:*
dtype0
?
Adam/decoder_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/decoder_output/kernel/v
?
0Adam/decoder_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decoder_output/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/conv2d_transpose/bias/v
?
0Adam/conv2d_transpose/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*/
shared_name Adam/conv2d_transpose/kernel/v
?
2Adam/conv2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/kernel/v*&
_output_shapes
: @*
dtype0

Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?b*$
shared_nameAdam/dense_3/bias/v
x
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes	
:?b*
dtype0
?
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?b*&
shared_nameAdam/dense_3/kernel/v
?
)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes
:	?b*
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

: *
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?b *$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	?b *
dtype0
?
Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_3/bias/v
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_3/kernel/v
?
*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_2/kernel/v
?
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_1/kernel/v
?
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
: @*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d/kernel/v
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/decoder_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/decoder_output/bias/m
?
.Adam/decoder_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/decoder_output/bias/m*
_output_shapes
:*
dtype0
?
Adam/decoder_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/decoder_output/kernel/m
?
0Adam/decoder_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decoder_output/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_nameAdam/conv2d_transpose/bias/m
?
0Adam/conv2d_transpose/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*/
shared_name Adam/conv2d_transpose/kernel/m
?
2Adam/conv2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/kernel/m*&
_output_shapes
: @*
dtype0

Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?b*$
shared_nameAdam/dense_3/bias/m
x
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes	
:?b*
dtype0
?
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?b*&
shared_nameAdam/dense_3/kernel/m
?
)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes
:	?b*
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

: *
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?b *$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	?b *
dtype0
?
Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_3/kernel/m
?
*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_2/kernel/m
?
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv2d_1/kernel/m
?
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
: @*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d/kernel/m
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
~
decoder_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namedecoder_output/bias
w
'decoder_output/bias/Read/ReadVariableOpReadVariableOpdecoder_output/bias*
_output_shapes
:*
dtype0
?
decoder_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_namedecoder_output/kernel
?
)decoder_output/kernel/Read/ReadVariableOpReadVariableOpdecoder_output/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameconv2d_transpose/bias
{
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes
: *
dtype0
?
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameconv2d_transpose/kernel
?
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*&
_output_shapes
: @*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?b*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:?b*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?b*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	?b*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

: *
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

: *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
: *
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?b *
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?b *
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:@*
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: @*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
: *
dtype0
?
serving_default_encoder_inputPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_encoder_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasconv2d_transpose/kernelconv2d_transpose/biasdecoder_output/kerneldecoder_output/biasConst_1Const*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_116394

NoOpNoOp
??
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer_with_weights-7
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures*
* 
?
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias
 (_jit_compiled_convolution_op*
?
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias
 1_jit_compiled_convolution_op*
?
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias
 :_jit_compiled_convolution_op*
?
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

Akernel
Bbias
 C_jit_compiled_convolution_op*
?
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses* 
?
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses

Pkernel
Qbias*
?
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses

Xkernel
Ybias*
?
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias*
?
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses* 
?
hlayer-0
ilayer_with_weights-0
ilayer-1
jlayer-2
klayer_with_weights-1
klayer-3
llayer_with_weights-2
llayer-4
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses*

s	keras_api* 

t	keras_api* 

u	keras_api* 

v	keras_api* 

w	keras_api* 

x	keras_api* 

y	keras_api* 

z	keras_api* 

{	keras_api* 
?
|	variables
}trainable_variables
~regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
&0
'1
/2
03
84
95
A6
B7
P8
Q9
X10
Y11
`12
a13
?14
?15
?16
?17
?18
?19*
?
&0
'1
/2
03
84
95
A6
B7
P8
Q9
X10
Y11
`12
a13
?14
?15
?16
?17
?18
?19*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
$
?
capture_20
?
capture_21* 
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate&m?'m?/m?0m?8m?9m?Am?Bm?Pm?Qm?Xm?Ym?`m?am?	?m?	?m?	?m?	?m?	?m?	?m?&v?'v?/v?0v?8v?9v?Av?Bv?Pv?Qv?Xv?Yv?`v?av?	?v?	?v?	?v?	?v?	?v?	?v?*
* 

?serving_default* 

&0
'1*

&0
'1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

/0
01*

/0
01*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

80
91*

80
91*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

A0
B1*

A0
B1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

P0
Q1*

P0
Q1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

X0
Y1*

X0
Y1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

`0
a1*

`0
a1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
!?_jit_compiled_convolution_op*
4
?0
?1
?2
?3
?4
?5*
4
?0
?1
?2
?3
?4
?5*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
|	variables
}trainable_variables
~regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
OI
VARIABLE_VALUEdense_3/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_3/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEconv2d_transpose/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEdecoder_output/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEdecoder_output/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
* 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20*

?0*
* 
* 
$
?
capture_20
?
capture_21* 
$
?
capture_20
?
capture_21* 
$
?
capture_20
?
capture_21* 
$
?
capture_20
?
capture_21* 
$
?
capture_20
?
capture_21* 
$
?
capture_20
?
capture_21* 
$
?
capture_20
?
capture_21* 
$
?
capture_20
?
capture_21* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
$
?
capture_20
?
capture_21* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
'
h0
i1
j2
k3
l4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
?	variables
?	keras_api

?total

?count*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_3/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_3/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/conv2d_transpose/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/decoder_output/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/decoder_output/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_3/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_3/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/conv2d_transpose/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/decoder_output/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/decoder_output/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp)decoder_output/kernel/Read/ReadVariableOp'decoder_output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp2Adam/conv2d_transpose/kernel/m/Read/ReadVariableOp0Adam/conv2d_transpose/bias/m/Read/ReadVariableOp0Adam/decoder_output/kernel/m/Read/ReadVariableOp.Adam/decoder_output/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp2Adam/conv2d_transpose/kernel/v/Read/ReadVariableOp0Adam/conv2d_transpose/bias/v/Read/ReadVariableOp0Adam/decoder_output/kernel/v/Read/ReadVariableOp.Adam/decoder_output/bias/v/Read/ReadVariableOpConst_2*P
TinI
G2E	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_117511
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasconv2d_transpose/kernelconv2d_transpose/biasdecoder_output/kerneldecoder_output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/conv2d_transpose/kernel/mAdam/conv2d_transpose/bias/mAdam/decoder_output/kernel/mAdam/decoder_output/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/vAdam/conv2d_transpose/kernel/vAdam/conv2d_transpose/bias/vAdam/decoder_output/kernel/vAdam/decoder_output/bias/v*O
TinH
F2D*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_117722??
?
q
B__inference_lambda_layer_call_and_return_conditional_losses_116973
inputs_0
inputs_1
identity?=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2????
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????F
ExpExpinputs_1*
T0*'
_output_shapes
:?????????X
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????Q
addAddV2inputs_0mul:z:0*
T0*'
_output_shapes
:?????????O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
C__inference_decoder_layer_call_and_return_conditional_losses_115405

inputs!
dense_3_115373:	?b
dense_3_115375:	?b1
conv2d_transpose_115394: @%
conv2d_transpose_115396: /
decoder_output_115399: #
decoder_output_115401:
identity??(conv2d_transpose/StatefulPartitionedCall?&decoder_output/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_115373dense_3_115375*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_115372?
reshape/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_115392?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_115394conv2d_transpose_115396*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_115302?
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0decoder_output_115399decoder_output_115401*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_115347?
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_add_loss_layer_call_fn_117155

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *\
_output_shapesJ
H:"??????????????????:"??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_add_loss_layer_call_and_return_conditional_losses_115751q
IdentityIdentityPartitionedCall:output:0*
T0*8
_output_shapes&
$:"??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:"??????????????????:` \
8
_output_shapes&
$:"??????????????????
 
_user_specified_nameinputs
?
p
'__inference_lambda_layer_call_fn_116953
inputs_0
inputs_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_115841o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
$__inference_vae_layer_call_fn_116494

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:	?b 
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12:

unknown_13:	?b

unknown_14:	?b$

unknown_15: @

unknown_16: $

unknown_17: 

unknown_18:

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:?????????:"??????????????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_vae_layer_call_and_return_conditional_losses_116061w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
/__inference_decoder_output_layer_call_fn_117251

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_115347?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
(__inference_decoder_layer_call_fn_117027

inputs
unknown:	?b
	unknown_0:	?b#
	unknown_1: @
	unknown_2: #
	unknown_3: 
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_115475w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_1_layer_call_fn_116821

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_115582w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
C__inference_dense_2_layer_call_and_return_conditional_losses_116941

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
(__inference_decoder_layer_call_fn_117010

inputs
unknown:	?b
	unknown_0:	?b#
	unknown_1: @
	unknown_2: #
	unknown_3: 
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_115405w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_dense_3_layer_call_fn_117169

inputs
unknown:	?b
	unknown_0:	?b
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_115372p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????b`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?[
?
?__inference_vae_layer_call_and_return_conditional_losses_116337
encoder_input'
conv2d_116251: 
conv2d_116253: )
conv2d_1_116256: @
conv2d_1_116258:@)
conv2d_2_116261:@@
conv2d_2_116263:@)
conv2d_3_116266:@@
conv2d_3_116268:@
dense_116272:	?b 
dense_116274:  
dense_1_116277: 
dense_1_116279: 
dense_2_116282: 
dense_2_116284:!
decoder_116288:	?b
decoder_116290:	?b(
decoder_116292: @
decoder_116294: (
decoder_116296: 
decoder_116298:
unknown
	unknown_0
identity

identity_1??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?decoder/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?lambda/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallencoder_inputconv2d_116251conv2d_116253*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_115565?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_116256conv2d_1_116258*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_115582?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_116261conv2d_2_116263*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_115599?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_116266conv2d_3_116268*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_115616?
flatten/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_115628?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_116272dense_116274*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_115641?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_116277dense_1_116279*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_115657?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_2_116282dense_2_116284*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_115673?
lambda/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_115841?
decoder/StatefulPartitionedCallStatefulPartitionedCall'lambda/StatefulPartitionedCall:output:0decoder_116288decoder_116290decoder_116292decoder_116294decoder_116296decoder_116298*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_115475o
*tf.keras.backend.binary_crossentropy/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *???3o
*tf.keras.backend.binary_crossentropy/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
(tf.keras.backend.binary_crossentropy/subSub3tf.keras.backend.binary_crossentropy/sub/x:output:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*
_output_shapes
: ?
:tf.keras.backend.binary_crossentropy/clip_by_value/MinimumMinimum(decoder/StatefulPartitionedCall:output:0,tf.keras.backend.binary_crossentropy/sub:z:0*
T0*/
_output_shapes
:??????????
2tf.keras.backend.binary_crossentropy/clip_by_valueMaximum>tf.keras.backend.binary_crossentropy/clip_by_value/Minimum:z:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*/
_output_shapes
:?????????o
*tf.keras.backend.binary_crossentropy/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
(tf.keras.backend.binary_crossentropy/addAddV26tf.keras.backend.binary_crossentropy/clip_by_value:z:03tf.keras.backend.binary_crossentropy/add/y:output:0*
T0*/
_output_shapes
:??????????
(tf.keras.backend.binary_crossentropy/LogLog,tf.keras.backend.binary_crossentropy/add:z:0*
T0*/
_output_shapes
:??????????
(tf.keras.backend.binary_crossentropy/mulMulencoder_input,tf.keras.backend.binary_crossentropy/Log:y:0*
T0*/
_output_shapes
:?????????q
,tf.keras.backend.binary_crossentropy/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
*tf.keras.backend.binary_crossentropy/sub_1Sub5tf.keras.backend.binary_crossentropy/sub_1/x:output:0encoder_input*
T0*/
_output_shapes
:?????????q
,tf.keras.backend.binary_crossentropy/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
*tf.keras.backend.binary_crossentropy/sub_2Sub5tf.keras.backend.binary_crossentropy/sub_2/x:output:06tf.keras.backend.binary_crossentropy/clip_by_value:z:0*
T0*/
_output_shapes
:?????????q
,tf.keras.backend.binary_crossentropy/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
*tf.keras.backend.binary_crossentropy/add_1AddV2.tf.keras.backend.binary_crossentropy/sub_2:z:05tf.keras.backend.binary_crossentropy/add_1/y:output:0*
T0*/
_output_shapes
:??????????
*tf.keras.backend.binary_crossentropy/Log_1Log.tf.keras.backend.binary_crossentropy/add_1:z:0*
T0*/
_output_shapes
:??????????
*tf.keras.backend.binary_crossentropy/mul_1Mul.tf.keras.backend.binary_crossentropy/sub_1:z:0.tf.keras.backend.binary_crossentropy/Log_1:y:0*
T0*/
_output_shapes
:??????????
*tf.keras.backend.binary_crossentropy/add_2AddV2,tf.keras.backend.binary_crossentropy/mul:z:0.tf.keras.backend.binary_crossentropy/mul_1:z:0*
T0*/
_output_shapes
:??????????
(tf.keras.backend.binary_crossentropy/NegNeg.tf.keras.backend.binary_crossentropy/add_2:z:0*
T0*/
_output_shapes
:?????????r
tf.math.exp/ExpExp(dense_2/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:??????????
tf.__operators__.add/AddV2AddV2unknown(dense_2/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????{
tf.math.square/SquareSquare(dense_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:??????????
tf.math.subtract/SubSubtf.__operators__.add/AddV2:z:0tf.math.square/Square:y:0*
T0*'
_output_shapes
:?????????~
tf.math.subtract_1/SubSubtf.math.subtract/Sub:z:0tf.math.exp/Exp:y:0*
T0*'
_output_shapes
:?????????u
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
tf.math.reduce_mean/MeanMeantf.math.subtract_1/Sub:z:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*#
_output_shapes
:?????????w
tf.math.multiply/MulMul	unknown_0!tf.math.reduce_mean/Mean:output:0*
T0*#
_output_shapes
:??????????
tf.__operators__.add_1/AddV2AddV2,tf.keras.backend.binary_crossentropy/Neg:y:0tf.math.multiply/Mul:z:0*
T0*8
_output_shapes&
$:"???????????????????
add_loss/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *\
_output_shapesJ
H:"??????????????????:"??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_add_loss_layer_call_and_return_conditional_losses_115751
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????

Identity_1Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*8
_output_shapes&
$:"???????????????????
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^decoder/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^lambda/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????: : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2@
lambda/StatefulPartitionedCalllambda/StatefulPartitionedCall:^ Z
/
_output_shapes
:?????????
'
_user_specified_nameencoder_input:

_output_shapes
: :

_output_shapes
: 
?
?
C__inference_decoder_layer_call_and_return_conditional_losses_115527
decoder_input!
dense_3_115510:	?b
dense_3_115512:	?b1
conv2d_transpose_115516: @%
conv2d_transpose_115518: /
decoder_output_115521: #
decoder_output_115523:
identity??(conv2d_transpose/StatefulPartitionedCall?&decoder_output/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCalldecoder_inputdense_3_115510dense_3_115512*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_115372?
reshape/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_115392?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_115516conv2d_transpose_115518*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_115302?
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0decoder_output_115521decoder_output_115523*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_115347?
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedecoder_input
?
?
)__inference_conv2d_2_layer_call_fn_116841

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_115599w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_116883

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????bY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????b"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
_
C__inference_reshape_layer_call_and_return_conditional_losses_115392

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????@`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????b:P L
(
_output_shapes
:??????????b
 
_user_specified_nameinputs
?
?
C__inference_decoder_layer_call_and_return_conditional_losses_115547
decoder_input!
dense_3_115530:	?b
dense_3_115532:	?b1
conv2d_transpose_115536: @%
conv2d_transpose_115538: /
decoder_output_115541: #
decoder_output_115543:
identity??(conv2d_transpose/StatefulPartitionedCall?&decoder_output/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCalldecoder_inputdense_3_115530dense_3_115532*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_115372?
reshape/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_115392?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_115536conv2d_transpose_115538*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_115302?
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0decoder_output_115541decoder_output_115543*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_115347?
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedecoder_input
?Z
?
?__inference_vae_layer_call_and_return_conditional_losses_116061

inputs'
conv2d_115975: 
conv2d_115977: )
conv2d_1_115980: @
conv2d_1_115982:@)
conv2d_2_115985:@@
conv2d_2_115987:@)
conv2d_3_115990:@@
conv2d_3_115992:@
dense_115996:	?b 
dense_115998:  
dense_1_116001: 
dense_1_116003: 
dense_2_116006: 
dense_2_116008:!
decoder_116012:	?b
decoder_116014:	?b(
decoder_116016: @
decoder_116018: (
decoder_116020: 
decoder_116022:
unknown
	unknown_0
identity

identity_1??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?decoder/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?lambda/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_115975conv2d_115977*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_115565?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_115980conv2d_1_115982*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_115582?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_115985conv2d_2_115987*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_115599?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_115990conv2d_3_115992*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_115616?
flatten/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_115628?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_115996dense_115998*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_115641?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_116001dense_1_116003*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_115657?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_2_116006dense_2_116008*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_115673?
lambda/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_115841?
decoder/StatefulPartitionedCallStatefulPartitionedCall'lambda/StatefulPartitionedCall:output:0decoder_116012decoder_116014decoder_116016decoder_116018decoder_116020decoder_116022*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_115475o
*tf.keras.backend.binary_crossentropy/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *???3o
*tf.keras.backend.binary_crossentropy/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
(tf.keras.backend.binary_crossentropy/subSub3tf.keras.backend.binary_crossentropy/sub/x:output:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*
_output_shapes
: ?
:tf.keras.backend.binary_crossentropy/clip_by_value/MinimumMinimum(decoder/StatefulPartitionedCall:output:0,tf.keras.backend.binary_crossentropy/sub:z:0*
T0*/
_output_shapes
:??????????
2tf.keras.backend.binary_crossentropy/clip_by_valueMaximum>tf.keras.backend.binary_crossentropy/clip_by_value/Minimum:z:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*/
_output_shapes
:?????????o
*tf.keras.backend.binary_crossentropy/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
(tf.keras.backend.binary_crossentropy/addAddV26tf.keras.backend.binary_crossentropy/clip_by_value:z:03tf.keras.backend.binary_crossentropy/add/y:output:0*
T0*/
_output_shapes
:??????????
(tf.keras.backend.binary_crossentropy/LogLog,tf.keras.backend.binary_crossentropy/add:z:0*
T0*/
_output_shapes
:??????????
(tf.keras.backend.binary_crossentropy/mulMulinputs,tf.keras.backend.binary_crossentropy/Log:y:0*
T0*/
_output_shapes
:?????????q
,tf.keras.backend.binary_crossentropy/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
*tf.keras.backend.binary_crossentropy/sub_1Sub5tf.keras.backend.binary_crossentropy/sub_1/x:output:0inputs*
T0*/
_output_shapes
:?????????q
,tf.keras.backend.binary_crossentropy/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
*tf.keras.backend.binary_crossentropy/sub_2Sub5tf.keras.backend.binary_crossentropy/sub_2/x:output:06tf.keras.backend.binary_crossentropy/clip_by_value:z:0*
T0*/
_output_shapes
:?????????q
,tf.keras.backend.binary_crossentropy/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
*tf.keras.backend.binary_crossentropy/add_1AddV2.tf.keras.backend.binary_crossentropy/sub_2:z:05tf.keras.backend.binary_crossentropy/add_1/y:output:0*
T0*/
_output_shapes
:??????????
*tf.keras.backend.binary_crossentropy/Log_1Log.tf.keras.backend.binary_crossentropy/add_1:z:0*
T0*/
_output_shapes
:??????????
*tf.keras.backend.binary_crossentropy/mul_1Mul.tf.keras.backend.binary_crossentropy/sub_1:z:0.tf.keras.backend.binary_crossentropy/Log_1:y:0*
T0*/
_output_shapes
:??????????
*tf.keras.backend.binary_crossentropy/add_2AddV2,tf.keras.backend.binary_crossentropy/mul:z:0.tf.keras.backend.binary_crossentropy/mul_1:z:0*
T0*/
_output_shapes
:??????????
(tf.keras.backend.binary_crossentropy/NegNeg.tf.keras.backend.binary_crossentropy/add_2:z:0*
T0*/
_output_shapes
:?????????r
tf.math.exp/ExpExp(dense_2/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:??????????
tf.__operators__.add/AddV2AddV2unknown(dense_2/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????{
tf.math.square/SquareSquare(dense_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:??????????
tf.math.subtract/SubSubtf.__operators__.add/AddV2:z:0tf.math.square/Square:y:0*
T0*'
_output_shapes
:?????????~
tf.math.subtract_1/SubSubtf.math.subtract/Sub:z:0tf.math.exp/Exp:y:0*
T0*'
_output_shapes
:?????????u
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
tf.math.reduce_mean/MeanMeantf.math.subtract_1/Sub:z:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*#
_output_shapes
:?????????w
tf.math.multiply/MulMul	unknown_0!tf.math.reduce_mean/Mean:output:0*
T0*#
_output_shapes
:??????????
tf.__operators__.add_1/AddV2AddV2,tf.keras.backend.binary_crossentropy/Neg:y:0tf.math.multiply/Mul:z:0*
T0*8
_output_shapes&
$:"???????????????????
add_loss/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *\
_output_shapesJ
H:"??????????????????:"??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_add_loss_layer_call_and_return_conditional_losses_115751
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????

Identity_1Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*8
_output_shapes&
$:"???????????????????
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^decoder/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^lambda/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????: : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2@
lambda/StatefulPartitionedCalllambda/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?	
?
C__inference_dense_1_layer_call_and_return_conditional_losses_116922

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
__inference__traced_save_117511
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop4
0savev2_decoder_output_kernel_read_readvariableop2
.savev2_decoder_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop=
9savev2_adam_conv2d_transpose_kernel_m_read_readvariableop;
7savev2_adam_conv2d_transpose_bias_m_read_readvariableop;
7savev2_adam_decoder_output_kernel_m_read_readvariableop9
5savev2_adam_decoder_output_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop=
9savev2_adam_conv2d_transpose_kernel_v_read_readvariableop;
7savev2_adam_conv2d_transpose_bias_v_read_readvariableop;
7savev2_adam_decoder_output_kernel_v_read_readvariableop9
5savev2_adam_decoder_output_bias_v_read_readvariableop
savev2_const_2

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?$
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?#
value?#B?#DB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?
value?B?DB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop0savev2_decoder_output_kernel_read_readvariableop.savev2_decoder_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop9savev2_adam_conv2d_transpose_kernel_m_read_readvariableop7savev2_adam_conv2d_transpose_bias_m_read_readvariableop7savev2_adam_decoder_output_kernel_m_read_readvariableop5savev2_adam_decoder_output_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop9savev2_adam_conv2d_transpose_kernel_v_read_readvariableop7savev2_adam_conv2d_transpose_bias_v_read_readvariableop7savev2_adam_decoder_output_kernel_v_read_readvariableop5savev2_adam_decoder_output_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *R
dtypesH
F2D	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : @:@:@@:@:@@:@:	?b : : :: ::	?b:?b: @: : :: : : : : : : : : : @:@:@@:@:@@:@:	?b : : :: ::	?b:?b: @: : :: : : @:@:@@:@:@@:@:	?b : : :: ::	?b:?b: @: : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:%	!

_output_shapes
:	?b : 


_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
::%!

_output_shapes
:	?b:!

_output_shapes	
:?b:,(
&
_output_shapes
: @: 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:, (
&
_output_shapes
:@@: !

_output_shapes
:@:,"(
&
_output_shapes
:@@: #

_output_shapes
:@:%$!

_output_shapes
:	?b : %

_output_shapes
: :$& 

_output_shapes

: : '

_output_shapes
::$( 

_output_shapes

: : )

_output_shapes
::%*!

_output_shapes
:	?b:!+

_output_shapes	
:?b:,,(
&
_output_shapes
: @: -

_output_shapes
: :,.(
&
_output_shapes
: : /

_output_shapes
::,0(
&
_output_shapes
: : 1

_output_shapes
: :,2(
&
_output_shapes
: @: 3

_output_shapes
:@:,4(
&
_output_shapes
:@@: 5

_output_shapes
:@:,6(
&
_output_shapes
:@@: 7

_output_shapes
:@:%8!

_output_shapes
:	?b : 9

_output_shapes
: :$: 

_output_shapes

: : ;

_output_shapes
::$< 

_output_shapes

: : =

_output_shapes
::%>!

_output_shapes
:	?b:!?

_output_shapes	
:?b:,@(
&
_output_shapes
: @: A

_output_shapes
: :,B(
&
_output_shapes
: : C

_output_shapes
::D

_output_shapes
: 
?
o
B__inference_lambda_layer_call_and_return_conditional_losses_115699

inputs
inputs_1
identity?;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2????
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????F
ExpExpinputs_1*
T0*'
_output_shapes
:?????????X
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????O
addAddV2inputsmul:z:0*
T0*'
_output_shapes
:?????????O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?I
?
C__inference_decoder_layer_call_and_return_conditional_losses_117149

inputs9
&dense_3_matmul_readvariableop_resource:	?b6
'dense_3_biasadd_readvariableop_resource:	?bS
9conv2d_transpose_conv2d_transpose_readvariableop_resource: @>
0conv2d_transpose_biasadd_readvariableop_resource: Q
7decoder_output_conv2d_transpose_readvariableop_resource: <
.decoder_output_biasadd_readvariableop_resource:
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?%decoder_output/BiasAdd/ReadVariableOp?.decoder_output/conv2d_transpose/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?b*
dtype0z
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????ba
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????bW
reshape/ShapeShapedense_3/Relu:activations:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshapedense_3/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@^
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? z
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:????????? g
decoder_output/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:l
"decoder_output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$decoder_output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$decoder_output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
decoder_output/strided_sliceStridedSlicedecoder_output/Shape:output:0+decoder_output/strided_slice/stack:output:0-decoder_output/strided_slice/stack_1:output:0-decoder_output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
decoder_output/stack/1Const*
_output_shapes
: *
dtype0*
value	B :X
decoder_output/stack/2Const*
_output_shapes
: *
dtype0*
value	B :X
decoder_output/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
decoder_output/stackPack%decoder_output/strided_slice:output:0decoder_output/stack/1:output:0decoder_output/stack/2:output:0decoder_output/stack/3:output:0*
N*
T0*
_output_shapes
:n
$decoder_output/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&decoder_output/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&decoder_output/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
decoder_output/strided_slice_1StridedSlicedecoder_output/stack:output:0-decoder_output/strided_slice_1/stack:output:0/decoder_output/strided_slice_1/stack_1:output:0/decoder_output/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
.decoder_output/conv2d_transpose/ReadVariableOpReadVariableOp7decoder_output_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
decoder_output/conv2d_transposeConv2DBackpropInputdecoder_output/stack:output:06decoder_output/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
%decoder_output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
decoder_output/BiasAddBiasAdd(decoder_output/conv2d_transpose:output:0-decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????|
decoder_output/SigmoidSigmoiddecoder_output/BiasAdd:output:0*
T0*/
_output_shapes
:?????????q
IdentityIdentitydecoder_output/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp&^decoder_output/BiasAdd/ReadVariableOp/^decoder_output/conv2d_transpose/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2N
%decoder_output/BiasAdd/ReadVariableOp%decoder_output/BiasAdd/ReadVariableOp2`
.decoder_output/conv2d_transpose/ReadVariableOp.decoder_output/conv2d_transpose/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
p
'__inference_lambda_layer_call_fn_116947
inputs_0
inputs_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_115699o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
(__inference_dense_2_layer_call_fn_116931

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_115673o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
C__inference_dense_1_layer_call_and_return_conditional_losses_115657

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?!
?
J__inference_decoder_output_layer_call_and_return_conditional_losses_117285

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
B__inference_conv2d_layer_call_and_return_conditional_losses_115565

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_117242

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_116852

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
C__inference_dense_3_layer_call_and_return_conditional_losses_115372

inputs1
matmul_readvariableop_resource:	?b.
biasadd_readvariableop_resource:	?b
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?b*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????bs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????bQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????bb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????bw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_decoder_layer_call_and_return_conditional_losses_115475

inputs!
dense_3_115458:	?b
dense_3_115460:	?b1
conv2d_transpose_115464: @%
conv2d_transpose_115466: /
decoder_output_115469: #
decoder_output_115471:
identity??(conv2d_transpose/StatefulPartitionedCall?&decoder_output/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_115458dense_3_115460*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_115372?
reshape/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_115392?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_115464conv2d_transpose_115466*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_115302?
&decoder_output/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0decoder_output_115469decoder_output_115471*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_decoder_output_layer_call_and_return_conditional_losses_115347?
IdentityIdentity/decoder_output/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp)^conv2d_transpose/StatefulPartitionedCall'^decoder_output/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2P
&decoder_output/StatefulPartitionedCall&decoder_output/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_115628

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????bY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????b"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
?__inference_vae_layer_call_and_return_conditional_losses_116792

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@@6
(conv2d_3_biasadd_readvariableop_resource:@7
$dense_matmul_readvariableop_resource:	?b 3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource: 5
'dense_2_biasadd_readvariableop_resource:A
.decoder_dense_3_matmul_readvariableop_resource:	?b>
/decoder_dense_3_biasadd_readvariableop_resource:	?b[
Adecoder_conv2d_transpose_conv2d_transpose_readvariableop_resource: @F
8decoder_conv2d_transpose_biasadd_readvariableop_resource: Y
?decoder_decoder_output_conv2d_transpose_readvariableop_resource: D
6decoder_decoder_output_biasadd_readvariableop_resource:
unknown
	unknown_0
identity

identity_1??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?/decoder/conv2d_transpose/BiasAdd/ReadVariableOp?8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp?-decoder/decoder_output/BiasAdd/ReadVariableOp?6decoder/decoder_output/conv2d_transpose/ReadVariableOp?&decoder/dense_3/BiasAdd/ReadVariableOp?%decoder/dense_3/MatMul/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@j
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  ?
flatten/ReshapeReshapeconv2d_3/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????b?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?b *
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? \

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_2/MatMulMatMuldense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T
lambda/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:d
lambda/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lambda/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lambda/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lambda/strided_sliceStridedSlicelambda/Shape:output:0#lambda/strided_slice/stack:output:0%lambda/strided_slice/stack_1:output:0%lambda/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
lambda/random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
lambda/random_normal/shapePacklambda/strided_slice:output:0%lambda/random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:^
lambda/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    `
lambda/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)lambda/random_normal/RandomStandardNormalRandomStandardNormal#lambda/random_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2???
lambda/random_normal/mulMul2lambda/random_normal/RandomStandardNormal:output:0$lambda/random_normal/stddev:output:0*
T0*'
_output_shapes
:??????????
lambda/random_normalAddV2lambda/random_normal/mul:z:0"lambda/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????]

lambda/ExpExpdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????m

lambda/mulMullambda/Exp:y:0lambda/random_normal:z:0*
T0*'
_output_shapes
:?????????o

lambda/addAddV2dense_1/BiasAdd:output:0lambda/mul:z:0*
T0*'
_output_shapes
:??????????
%decoder/dense_3/MatMul/ReadVariableOpReadVariableOp.decoder_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?b*
dtype0?
decoder/dense_3/MatMulMatMullambda/add:z:0-decoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b?
&decoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype0?
decoder/dense_3/BiasAddBiasAdd decoder/dense_3/MatMul:product:0.decoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????bq
decoder/dense_3/ReluRelu decoder/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????bg
decoder/reshape/ShapeShape"decoder/dense_3/Relu:activations:0*
T0*
_output_shapes
:m
#decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
decoder/reshape/strided_sliceStridedSlicedecoder/reshape/Shape:output:0,decoder/reshape/strided_slice/stack:output:0.decoder/reshape/strided_slice/stack_1:output:0.decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :a
decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :a
decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@?
decoder/reshape/Reshape/shapePack&decoder/reshape/strided_slice:output:0(decoder/reshape/Reshape/shape/1:output:0(decoder/reshape/Reshape/shape/2:output:0(decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
decoder/reshape/ReshapeReshape"decoder/dense_3/Relu:activations:0&decoder/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@n
decoder/conv2d_transpose/ShapeShape decoder/reshape/Reshape:output:0*
T0*
_output_shapes
:v
,decoder/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.decoder/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.decoder/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&decoder/conv2d_transpose/strided_sliceStridedSlice'decoder/conv2d_transpose/Shape:output:05decoder/conv2d_transpose/strided_slice/stack:output:07decoder/conv2d_transpose/strided_slice/stack_1:output:07decoder/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 decoder/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b
 decoder/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :b
 decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
decoder/conv2d_transpose/stackPack/decoder/conv2d_transpose/strided_slice:output:0)decoder/conv2d_transpose/stack/1:output:0)decoder/conv2d_transpose/stack/2:output:0)decoder/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:x
.decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/conv2d_transpose/strided_slice_1StridedSlice'decoder/conv2d_transpose/stack:output:07decoder/conv2d_transpose/strided_slice_1/stack:output:09decoder/conv2d_transpose/strided_slice_1/stack_1:output:09decoder/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpAdecoder_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
)decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInput'decoder/conv2d_transpose/stack:output:0@decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0 decoder/reshape/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
/decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp8decoder_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
 decoder/conv2d_transpose/BiasAddBiasAdd2decoder/conv2d_transpose/conv2d_transpose:output:07decoder/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
decoder/conv2d_transpose/ReluRelu)decoder/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:????????? w
decoder/decoder_output/ShapeShape+decoder/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:t
*decoder/decoder_output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,decoder/decoder_output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,decoder/decoder_output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$decoder/decoder_output/strided_sliceStridedSlice%decoder/decoder_output/Shape:output:03decoder/decoder_output/strided_slice/stack:output:05decoder/decoder_output/strided_slice/stack_1:output:05decoder/decoder_output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
decoder/decoder_output/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`
decoder/decoder_output/stack/2Const*
_output_shapes
: *
dtype0*
value	B :`
decoder/decoder_output/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
decoder/decoder_output/stackPack-decoder/decoder_output/strided_slice:output:0'decoder/decoder_output/stack/1:output:0'decoder/decoder_output/stack/2:output:0'decoder/decoder_output/stack/3:output:0*
N*
T0*
_output_shapes
:v
,decoder/decoder_output/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.decoder/decoder_output/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.decoder/decoder_output/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&decoder/decoder_output/strided_slice_1StridedSlice%decoder/decoder_output/stack:output:05decoder/decoder_output/strided_slice_1/stack:output:07decoder/decoder_output/strided_slice_1/stack_1:output:07decoder/decoder_output/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
6decoder/decoder_output/conv2d_transpose/ReadVariableOpReadVariableOp?decoder_decoder_output_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
'decoder/decoder_output/conv2d_transposeConv2DBackpropInput%decoder/decoder_output/stack:output:0>decoder/decoder_output/conv2d_transpose/ReadVariableOp:value:0+decoder/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
-decoder/decoder_output/BiasAdd/ReadVariableOpReadVariableOp6decoder_decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
decoder/decoder_output/BiasAddBiasAdd0decoder/decoder_output/conv2d_transpose:output:05decoder/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
decoder/decoder_output/SigmoidSigmoid'decoder/decoder_output/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
=tf.keras.backend.binary_crossentropy/logistic_loss/zeros_like	ZerosLike'decoder/decoder_output/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
?tf.keras.backend.binary_crossentropy/logistic_loss/GreaterEqualGreaterEqual'decoder/decoder_output/BiasAdd:output:0Atf.keras.backend.binary_crossentropy/logistic_loss/zeros_like:y:0*
T0*/
_output_shapes
:??????????
9tf.keras.backend.binary_crossentropy/logistic_loss/SelectSelectCtf.keras.backend.binary_crossentropy/logistic_loss/GreaterEqual:z:0'decoder/decoder_output/BiasAdd:output:0Atf.keras.backend.binary_crossentropy/logistic_loss/zeros_like:y:0*
T0*/
_output_shapes
:??????????
6tf.keras.backend.binary_crossentropy/logistic_loss/NegNeg'decoder/decoder_output/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
;tf.keras.backend.binary_crossentropy/logistic_loss/Select_1SelectCtf.keras.backend.binary_crossentropy/logistic_loss/GreaterEqual:z:0:tf.keras.backend.binary_crossentropy/logistic_loss/Neg:y:0'decoder/decoder_output/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
6tf.keras.backend.binary_crossentropy/logistic_loss/mulMul'decoder/decoder_output/BiasAdd:output:0inputs*
T0*/
_output_shapes
:??????????
6tf.keras.backend.binary_crossentropy/logistic_loss/subSubBtf.keras.backend.binary_crossentropy/logistic_loss/Select:output:0:tf.keras.backend.binary_crossentropy/logistic_loss/mul:z:0*
T0*/
_output_shapes
:??????????
6tf.keras.backend.binary_crossentropy/logistic_loss/ExpExpDtf.keras.backend.binary_crossentropy/logistic_loss/Select_1:output:0*
T0*/
_output_shapes
:??????????
8tf.keras.backend.binary_crossentropy/logistic_loss/Log1pLog1p:tf.keras.backend.binary_crossentropy/logistic_loss/Exp:y:0*
T0*/
_output_shapes
:??????????
2tf.keras.backend.binary_crossentropy/logistic_lossAddV2:tf.keras.backend.binary_crossentropy/logistic_loss/sub:z:0<tf.keras.backend.binary_crossentropy/logistic_loss/Log1p:y:0*
T0*/
_output_shapes
:?????????b
tf.math.exp/ExpExpdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x
tf.__operators__.add/AddV2AddV2unknowndense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????k
tf.math.square/SquareSquaredense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
tf.math.subtract/SubSubtf.__operators__.add/AddV2:z:0tf.math.square/Square:y:0*
T0*'
_output_shapes
:?????????~
tf.math.subtract_1/SubSubtf.math.subtract/Sub:z:0tf.math.exp/Exp:y:0*
T0*'
_output_shapes
:?????????u
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
tf.math.reduce_mean/MeanMeantf.math.subtract_1/Sub:z:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*#
_output_shapes
:?????????w
tf.math.multiply/MulMul	unknown_0!tf.math.reduce_mean/Mean:output:0*
T0*#
_output_shapes
:??????????
tf.__operators__.add_1/AddV2AddV26tf.keras.backend.binary_crossentropy/logistic_loss:z:0tf.math.multiply/Mul:z:0*
T0*8
_output_shapes&
$:"??????????????????y
IdentityIdentity"decoder/decoder_output/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????

Identity_1Identity tf.__operators__.add_1/AddV2:z:0^NoOp*
T0*8
_output_shapes&
$:"???????????????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp0^decoder/conv2d_transpose/BiasAdd/ReadVariableOp9^decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp.^decoder/decoder_output/BiasAdd/ReadVariableOp7^decoder/decoder_output/conv2d_transpose/ReadVariableOp'^decoder/dense_3/BiasAdd/ReadVariableOp&^decoder/dense_3/MatMul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????: : : : : : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2b
/decoder/conv2d_transpose/BiasAdd/ReadVariableOp/decoder/conv2d_transpose/BiasAdd/ReadVariableOp2t
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2^
-decoder/decoder_output/BiasAdd/ReadVariableOp-decoder/decoder_output/BiasAdd/ReadVariableOp2p
6decoder/decoder_output/conv2d_transpose/ReadVariableOp6decoder/decoder_output/conv2d_transpose/ReadVariableOp2P
&decoder/dense_3/BiasAdd/ReadVariableOp&decoder/dense_3/BiasAdd/ReadVariableOp2N
%decoder/dense_3/MatMul/ReadVariableOp%decoder/dense_3/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
??
?
?__inference_vae_layer_call_and_return_conditional_losses_116643

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@A
'conv2d_2_conv2d_readvariableop_resource:@@6
(conv2d_2_biasadd_readvariableop_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@@6
(conv2d_3_biasadd_readvariableop_resource:@7
$dense_matmul_readvariableop_resource:	?b 3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource: 5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource: 5
'dense_2_biasadd_readvariableop_resource:A
.decoder_dense_3_matmul_readvariableop_resource:	?b>
/decoder_dense_3_biasadd_readvariableop_resource:	?b[
Adecoder_conv2d_transpose_conv2d_transpose_readvariableop_resource: @F
8decoder_conv2d_transpose_biasadd_readvariableop_resource: Y
?decoder_decoder_output_conv2d_transpose_readvariableop_resource: D
6decoder_decoder_output_biasadd_readvariableop_resource:
unknown
	unknown_0
identity

identity_1??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?/decoder/conv2d_transpose/BiasAdd/ReadVariableOp?8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp?-decoder/decoder_output/BiasAdd/ReadVariableOp?6decoder/decoder_output/conv2d_transpose/ReadVariableOp?&decoder/dense_3/BiasAdd/ReadVariableOp?%decoder/dense_3/MatMul/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@j
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  ?
flatten/ReshapeReshapeconv2d_3/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????b?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?b *
dtype0?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? \

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
dense_2/MatMulMatMuldense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????T
lambda/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:d
lambda/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lambda/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lambda/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
lambda/strided_sliceStridedSlicelambda/Shape:output:0#lambda/strided_slice/stack:output:0%lambda/strided_slice/stack_1:output:0%lambda/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
lambda/random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
lambda/random_normal/shapePacklambda/strided_slice:output:0%lambda/random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:^
lambda/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    `
lambda/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
)lambda/random_normal/RandomStandardNormalRandomStandardNormal#lambda/random_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2????
lambda/random_normal/mulMul2lambda/random_normal/RandomStandardNormal:output:0$lambda/random_normal/stddev:output:0*
T0*'
_output_shapes
:??????????
lambda/random_normalAddV2lambda/random_normal/mul:z:0"lambda/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????]

lambda/ExpExpdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????m

lambda/mulMullambda/Exp:y:0lambda/random_normal:z:0*
T0*'
_output_shapes
:?????????o

lambda/addAddV2dense_1/BiasAdd:output:0lambda/mul:z:0*
T0*'
_output_shapes
:??????????
%decoder/dense_3/MatMul/ReadVariableOpReadVariableOp.decoder_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?b*
dtype0?
decoder/dense_3/MatMulMatMullambda/add:z:0-decoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b?
&decoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype0?
decoder/dense_3/BiasAddBiasAdd decoder/dense_3/MatMul:product:0.decoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????bq
decoder/dense_3/ReluRelu decoder/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????bg
decoder/reshape/ShapeShape"decoder/dense_3/Relu:activations:0*
T0*
_output_shapes
:m
#decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
decoder/reshape/strided_sliceStridedSlicedecoder/reshape/Shape:output:0,decoder/reshape/strided_slice/stack:output:0.decoder/reshape/strided_slice/stack_1:output:0.decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :a
decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :a
decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@?
decoder/reshape/Reshape/shapePack&decoder/reshape/strided_slice:output:0(decoder/reshape/Reshape/shape/1:output:0(decoder/reshape/Reshape/shape/2:output:0(decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
decoder/reshape/ReshapeReshape"decoder/dense_3/Relu:activations:0&decoder/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@n
decoder/conv2d_transpose/ShapeShape decoder/reshape/Reshape:output:0*
T0*
_output_shapes
:v
,decoder/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.decoder/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.decoder/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&decoder/conv2d_transpose/strided_sliceStridedSlice'decoder/conv2d_transpose/Shape:output:05decoder/conv2d_transpose/strided_slice/stack:output:07decoder/conv2d_transpose/strided_slice/stack_1:output:07decoder/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 decoder/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :b
 decoder/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :b
 decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
decoder/conv2d_transpose/stackPack/decoder/conv2d_transpose/strided_slice:output:0)decoder/conv2d_transpose/stack/1:output:0)decoder/conv2d_transpose/stack/2:output:0)decoder/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:x
.decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0decoder/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(decoder/conv2d_transpose/strided_slice_1StridedSlice'decoder/conv2d_transpose/stack:output:07decoder/conv2d_transpose/strided_slice_1/stack:output:09decoder/conv2d_transpose/strided_slice_1/stack_1:output:09decoder/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpAdecoder_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
)decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInput'decoder/conv2d_transpose/stack:output:0@decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0 decoder/reshape/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
/decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp8decoder_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
 decoder/conv2d_transpose/BiasAddBiasAdd2decoder/conv2d_transpose/conv2d_transpose:output:07decoder/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
decoder/conv2d_transpose/ReluRelu)decoder/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:????????? w
decoder/decoder_output/ShapeShape+decoder/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:t
*decoder/decoder_output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,decoder/decoder_output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,decoder/decoder_output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
$decoder/decoder_output/strided_sliceStridedSlice%decoder/decoder_output/Shape:output:03decoder/decoder_output/strided_slice/stack:output:05decoder/decoder_output/strided_slice/stack_1:output:05decoder/decoder_output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
decoder/decoder_output/stack/1Const*
_output_shapes
: *
dtype0*
value	B :`
decoder/decoder_output/stack/2Const*
_output_shapes
: *
dtype0*
value	B :`
decoder/decoder_output/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
decoder/decoder_output/stackPack-decoder/decoder_output/strided_slice:output:0'decoder/decoder_output/stack/1:output:0'decoder/decoder_output/stack/2:output:0'decoder/decoder_output/stack/3:output:0*
N*
T0*
_output_shapes
:v
,decoder/decoder_output/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.decoder/decoder_output/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.decoder/decoder_output/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&decoder/decoder_output/strided_slice_1StridedSlice%decoder/decoder_output/stack:output:05decoder/decoder_output/strided_slice_1/stack:output:07decoder/decoder_output/strided_slice_1/stack_1:output:07decoder/decoder_output/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
6decoder/decoder_output/conv2d_transpose/ReadVariableOpReadVariableOp?decoder_decoder_output_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
'decoder/decoder_output/conv2d_transposeConv2DBackpropInput%decoder/decoder_output/stack:output:0>decoder/decoder_output/conv2d_transpose/ReadVariableOp:value:0+decoder/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
-decoder/decoder_output/BiasAdd/ReadVariableOpReadVariableOp6decoder_decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
decoder/decoder_output/BiasAddBiasAdd0decoder/decoder_output/conv2d_transpose:output:05decoder/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
decoder/decoder_output/SigmoidSigmoid'decoder/decoder_output/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
=tf.keras.backend.binary_crossentropy/logistic_loss/zeros_like	ZerosLike'decoder/decoder_output/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
?tf.keras.backend.binary_crossentropy/logistic_loss/GreaterEqualGreaterEqual'decoder/decoder_output/BiasAdd:output:0Atf.keras.backend.binary_crossentropy/logistic_loss/zeros_like:y:0*
T0*/
_output_shapes
:??????????
9tf.keras.backend.binary_crossentropy/logistic_loss/SelectSelectCtf.keras.backend.binary_crossentropy/logistic_loss/GreaterEqual:z:0'decoder/decoder_output/BiasAdd:output:0Atf.keras.backend.binary_crossentropy/logistic_loss/zeros_like:y:0*
T0*/
_output_shapes
:??????????
6tf.keras.backend.binary_crossentropy/logistic_loss/NegNeg'decoder/decoder_output/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
;tf.keras.backend.binary_crossentropy/logistic_loss/Select_1SelectCtf.keras.backend.binary_crossentropy/logistic_loss/GreaterEqual:z:0:tf.keras.backend.binary_crossentropy/logistic_loss/Neg:y:0'decoder/decoder_output/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
6tf.keras.backend.binary_crossentropy/logistic_loss/mulMul'decoder/decoder_output/BiasAdd:output:0inputs*
T0*/
_output_shapes
:??????????
6tf.keras.backend.binary_crossentropy/logistic_loss/subSubBtf.keras.backend.binary_crossentropy/logistic_loss/Select:output:0:tf.keras.backend.binary_crossentropy/logistic_loss/mul:z:0*
T0*/
_output_shapes
:??????????
6tf.keras.backend.binary_crossentropy/logistic_loss/ExpExpDtf.keras.backend.binary_crossentropy/logistic_loss/Select_1:output:0*
T0*/
_output_shapes
:??????????
8tf.keras.backend.binary_crossentropy/logistic_loss/Log1pLog1p:tf.keras.backend.binary_crossentropy/logistic_loss/Exp:y:0*
T0*/
_output_shapes
:??????????
2tf.keras.backend.binary_crossentropy/logistic_lossAddV2:tf.keras.backend.binary_crossentropy/logistic_loss/sub:z:0<tf.keras.backend.binary_crossentropy/logistic_loss/Log1p:y:0*
T0*/
_output_shapes
:?????????b
tf.math.exp/ExpExpdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????x
tf.__operators__.add/AddV2AddV2unknowndense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????k
tf.math.square/SquareSquaredense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
tf.math.subtract/SubSubtf.__operators__.add/AddV2:z:0tf.math.square/Square:y:0*
T0*'
_output_shapes
:?????????~
tf.math.subtract_1/SubSubtf.math.subtract/Sub:z:0tf.math.exp/Exp:y:0*
T0*'
_output_shapes
:?????????u
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
tf.math.reduce_mean/MeanMeantf.math.subtract_1/Sub:z:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*#
_output_shapes
:?????????w
tf.math.multiply/MulMul	unknown_0!tf.math.reduce_mean/Mean:output:0*
T0*#
_output_shapes
:??????????
tf.__operators__.add_1/AddV2AddV26tf.keras.backend.binary_crossentropy/logistic_loss:z:0tf.math.multiply/Mul:z:0*
T0*8
_output_shapes&
$:"??????????????????y
IdentityIdentity"decoder/decoder_output/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????

Identity_1Identity tf.__operators__.add_1/AddV2:z:0^NoOp*
T0*8
_output_shapes&
$:"???????????????????
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp0^decoder/conv2d_transpose/BiasAdd/ReadVariableOp9^decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp.^decoder/decoder_output/BiasAdd/ReadVariableOp7^decoder/decoder_output/conv2d_transpose/ReadVariableOp'^decoder/dense_3/BiasAdd/ReadVariableOp&^decoder/dense_3/MatMul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????: : : : : : : : : : : : : : : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2b
/decoder/conv2d_transpose/BiasAdd/ReadVariableOp/decoder/conv2d_transpose/BiasAdd/ReadVariableOp2t
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2^
-decoder/decoder_output/BiasAdd/ReadVariableOp-decoder/decoder_output/BiasAdd/ReadVariableOp2p
6decoder/decoder_output/conv2d_transpose/ReadVariableOp6decoder/decoder_output/conv2d_transpose/ReadVariableOp2P
&decoder/dense_3/BiasAdd/ReadVariableOp&decoder/dense_3/BiasAdd/ReadVariableOp2N
%decoder/dense_3/MatMul/ReadVariableOp%decoder/dense_3/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
&__inference_dense_layer_call_fn_116892

inputs
unknown:	?b 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_115641o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????b: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????b
 
_user_specified_nameinputs
?!
?
J__inference_decoder_output_layer_call_and_return_conditional_losses_115347

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_115599

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_116832

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
1__inference_conv2d_transpose_layer_call_fn_117208

inputs!
unknown: @
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_115302?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
D
(__inference_reshape_layer_call_fn_117185

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_115392h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????b:P L
(
_output_shapes
:??????????b
 
_user_specified_nameinputs
?
D
(__inference_flatten_layer_call_fn_116877

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_115628a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????b"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
B__inference_conv2d_layer_call_and_return_conditional_losses_116812

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?I
?
C__inference_decoder_layer_call_and_return_conditional_losses_117088

inputs9
&dense_3_matmul_readvariableop_resource:	?b6
'dense_3_biasadd_readvariableop_resource:	?bS
9conv2d_transpose_conv2d_transpose_readvariableop_resource: @>
0conv2d_transpose_biasadd_readvariableop_resource: Q
7decoder_output_conv2d_transpose_readvariableop_resource: <
.decoder_output_biasadd_readvariableop_resource:
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?%decoder_output/BiasAdd/ReadVariableOp?.decoder_output/conv2d_transpose/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	?b*
dtype0z
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype0?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????ba
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????bW
reshape/ShapeShapedense_3/Relu:activations:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshapedense_3/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@^
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? z
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:????????? g
decoder_output/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:l
"decoder_output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$decoder_output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$decoder_output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
decoder_output/strided_sliceStridedSlicedecoder_output/Shape:output:0+decoder_output/strided_slice/stack:output:0-decoder_output/strided_slice/stack_1:output:0-decoder_output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
decoder_output/stack/1Const*
_output_shapes
: *
dtype0*
value	B :X
decoder_output/stack/2Const*
_output_shapes
: *
dtype0*
value	B :X
decoder_output/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
decoder_output/stackPack%decoder_output/strided_slice:output:0decoder_output/stack/1:output:0decoder_output/stack/2:output:0decoder_output/stack/3:output:0*
N*
T0*
_output_shapes
:n
$decoder_output/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&decoder_output/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&decoder_output/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
decoder_output/strided_slice_1StridedSlicedecoder_output/stack:output:0-decoder_output/strided_slice_1/stack:output:0/decoder_output/strided_slice_1/stack_1:output:0/decoder_output/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
.decoder_output/conv2d_transpose/ReadVariableOpReadVariableOp7decoder_output_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
decoder_output/conv2d_transposeConv2DBackpropInputdecoder_output/stack:output:06decoder_output/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
%decoder_output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
decoder_output/BiasAddBiasAdd(decoder_output/conv2d_transpose:output:0-decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????|
decoder_output/SigmoidSigmoiddecoder_output/BiasAdd:output:0*
T0*/
_output_shapes
:?????????q
IdentityIdentitydecoder_output/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp&^decoder_output/BiasAdd/ReadVariableOp/^decoder_output/conv2d_transpose/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2N
%decoder_output/BiasAdd/ReadVariableOp%decoder_output/BiasAdd/ReadVariableOp2`
.decoder_output/conv2d_transpose/ReadVariableOp.decoder_output/conv2d_transpose/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_115264
encoder_inputC
)vae_conv2d_conv2d_readvariableop_resource: 8
*vae_conv2d_biasadd_readvariableop_resource: E
+vae_conv2d_1_conv2d_readvariableop_resource: @:
,vae_conv2d_1_biasadd_readvariableop_resource:@E
+vae_conv2d_2_conv2d_readvariableop_resource:@@:
,vae_conv2d_2_biasadd_readvariableop_resource:@E
+vae_conv2d_3_conv2d_readvariableop_resource:@@:
,vae_conv2d_3_biasadd_readvariableop_resource:@;
(vae_dense_matmul_readvariableop_resource:	?b 7
)vae_dense_biasadd_readvariableop_resource: <
*vae_dense_1_matmul_readvariableop_resource: 9
+vae_dense_1_biasadd_readvariableop_resource:<
*vae_dense_2_matmul_readvariableop_resource: 9
+vae_dense_2_biasadd_readvariableop_resource:E
2vae_decoder_dense_3_matmul_readvariableop_resource:	?bB
3vae_decoder_dense_3_biasadd_readvariableop_resource:	?b_
Evae_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resource: @J
<vae_decoder_conv2d_transpose_biasadd_readvariableop_resource: ]
Cvae_decoder_decoder_output_conv2d_transpose_readvariableop_resource: H
:vae_decoder_decoder_output_biasadd_readvariableop_resource:

vae_115251

vae_115259
identity??!vae/conv2d/BiasAdd/ReadVariableOp? vae/conv2d/Conv2D/ReadVariableOp?#vae/conv2d_1/BiasAdd/ReadVariableOp?"vae/conv2d_1/Conv2D/ReadVariableOp?#vae/conv2d_2/BiasAdd/ReadVariableOp?"vae/conv2d_2/Conv2D/ReadVariableOp?#vae/conv2d_3/BiasAdd/ReadVariableOp?"vae/conv2d_3/Conv2D/ReadVariableOp?3vae/decoder/conv2d_transpose/BiasAdd/ReadVariableOp?<vae/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp?1vae/decoder/decoder_output/BiasAdd/ReadVariableOp?:vae/decoder/decoder_output/conv2d_transpose/ReadVariableOp?*vae/decoder/dense_3/BiasAdd/ReadVariableOp?)vae/decoder/dense_3/MatMul/ReadVariableOp? vae/dense/BiasAdd/ReadVariableOp?vae/dense/MatMul/ReadVariableOp?"vae/dense_1/BiasAdd/ReadVariableOp?!vae/dense_1/MatMul/ReadVariableOp?"vae/dense_2/BiasAdd/ReadVariableOp?!vae/dense_2/MatMul/ReadVariableOp?
 vae/conv2d/Conv2D/ReadVariableOpReadVariableOp)vae_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
vae/conv2d/Conv2DConv2Dencoder_input(vae/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
!vae/conv2d/BiasAdd/ReadVariableOpReadVariableOp*vae_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
vae/conv2d/BiasAddBiasAddvae/conv2d/Conv2D:output:0)vae/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? n
vae/conv2d/ReluReluvae/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
"vae/conv2d_1/Conv2D/ReadVariableOpReadVariableOp+vae_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
vae/conv2d_1/Conv2DConv2Dvae/conv2d/Relu:activations:0*vae/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
#vae/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp,vae_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
vae/conv2d_1/BiasAddBiasAddvae/conv2d_1/Conv2D:output:0+vae/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@r
vae/conv2d_1/ReluReluvae/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
"vae/conv2d_2/Conv2D/ReadVariableOpReadVariableOp+vae_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
vae/conv2d_2/Conv2DConv2Dvae/conv2d_1/Relu:activations:0*vae/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
#vae/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp,vae_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
vae/conv2d_2/BiasAddBiasAddvae/conv2d_2/Conv2D:output:0+vae/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@r
vae/conv2d_2/ReluReluvae/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
"vae/conv2d_3/Conv2D/ReadVariableOpReadVariableOp+vae_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
vae/conv2d_3/Conv2DConv2Dvae/conv2d_2/Relu:activations:0*vae/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
#vae/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp,vae_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
vae/conv2d_3/BiasAddBiasAddvae/conv2d_3/Conv2D:output:0+vae/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@r
vae/conv2d_3/ReluReluvae/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@b
vae/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  ?
vae/flatten/ReshapeReshapevae/conv2d_3/Relu:activations:0vae/flatten/Const:output:0*
T0*(
_output_shapes
:??????????b?
vae/dense/MatMul/ReadVariableOpReadVariableOp(vae_dense_matmul_readvariableop_resource*
_output_shapes
:	?b *
dtype0?
vae/dense/MatMulMatMulvae/flatten/Reshape:output:0'vae/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? ?
 vae/dense/BiasAdd/ReadVariableOpReadVariableOp)vae_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
vae/dense/BiasAddBiasAddvae/dense/MatMul:product:0(vae/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? d
vae/dense/ReluReluvae/dense/BiasAdd:output:0*
T0*'
_output_shapes
:????????? ?
!vae/dense_1/MatMul/ReadVariableOpReadVariableOp*vae_dense_1_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
vae/dense_1/MatMulMatMulvae/dense/Relu:activations:0)vae/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
"vae/dense_1/BiasAdd/ReadVariableOpReadVariableOp+vae_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
vae/dense_1/BiasAddBiasAddvae/dense_1/MatMul:product:0*vae/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!vae/dense_2/MatMul/ReadVariableOpReadVariableOp*vae_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype0?
vae/dense_2/MatMulMatMulvae/dense/Relu:activations:0)vae/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
"vae/dense_2/BiasAdd/ReadVariableOpReadVariableOp+vae_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
vae/dense_2/BiasAddBiasAddvae/dense_2/MatMul:product:0*vae/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????\
vae/lambda/ShapeShapevae/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:h
vae/lambda/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: j
 vae/lambda/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j
 vae/lambda/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
vae/lambda/strided_sliceStridedSlicevae/lambda/Shape:output:0'vae/lambda/strided_slice/stack:output:0)vae/lambda/strided_slice/stack_1:output:0)vae/lambda/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 vae/lambda/random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
vae/lambda/random_normal/shapePack!vae/lambda/strided_slice:output:0)vae/lambda/random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:b
vae/lambda/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    d
vae/lambda/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
-vae/lambda/random_normal/RandomStandardNormalRandomStandardNormal'vae/lambda/random_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2????
vae/lambda/random_normal/mulMul6vae/lambda/random_normal/RandomStandardNormal:output:0(vae/lambda/random_normal/stddev:output:0*
T0*'
_output_shapes
:??????????
vae/lambda/random_normalAddV2 vae/lambda/random_normal/mul:z:0&vae/lambda/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????e
vae/lambda/ExpExpvae/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????y
vae/lambda/mulMulvae/lambda/Exp:y:0vae/lambda/random_normal:z:0*
T0*'
_output_shapes
:?????????{
vae/lambda/addAddV2vae/dense_1/BiasAdd:output:0vae/lambda/mul:z:0*
T0*'
_output_shapes
:??????????
)vae/decoder/dense_3/MatMul/ReadVariableOpReadVariableOp2vae_decoder_dense_3_matmul_readvariableop_resource*
_output_shapes
:	?b*
dtype0?
vae/decoder/dense_3/MatMulMatMulvae/lambda/add:z:01vae/decoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b?
*vae/decoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp3vae_decoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype0?
vae/decoder/dense_3/BiasAddBiasAdd$vae/decoder/dense_3/MatMul:product:02vae/decoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????by
vae/decoder/dense_3/ReluRelu$vae/decoder/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????bo
vae/decoder/reshape/ShapeShape&vae/decoder/dense_3/Relu:activations:0*
T0*
_output_shapes
:q
'vae/decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)vae/decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)vae/decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!vae/decoder/reshape/strided_sliceStridedSlice"vae/decoder/reshape/Shape:output:00vae/decoder/reshape/strided_slice/stack:output:02vae/decoder/reshape/strided_slice/stack_1:output:02vae/decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#vae/decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :e
#vae/decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :e
#vae/decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@?
!vae/decoder/reshape/Reshape/shapePack*vae/decoder/reshape/strided_slice:output:0,vae/decoder/reshape/Reshape/shape/1:output:0,vae/decoder/reshape/Reshape/shape/2:output:0,vae/decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
vae/decoder/reshape/ReshapeReshape&vae/decoder/dense_3/Relu:activations:0*vae/decoder/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@v
"vae/decoder/conv2d_transpose/ShapeShape$vae/decoder/reshape/Reshape:output:0*
T0*
_output_shapes
:z
0vae/decoder/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2vae/decoder/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2vae/decoder/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*vae/decoder/conv2d_transpose/strided_sliceStridedSlice+vae/decoder/conv2d_transpose/Shape:output:09vae/decoder/conv2d_transpose/strided_slice/stack:output:0;vae/decoder/conv2d_transpose/strided_slice/stack_1:output:0;vae/decoder/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$vae/decoder/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :f
$vae/decoder/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :f
$vae/decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
"vae/decoder/conv2d_transpose/stackPack3vae/decoder/conv2d_transpose/strided_slice:output:0-vae/decoder/conv2d_transpose/stack/1:output:0-vae/decoder/conv2d_transpose/stack/2:output:0-vae/decoder/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:|
2vae/decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4vae/decoder/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4vae/decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,vae/decoder/conv2d_transpose/strided_slice_1StridedSlice+vae/decoder/conv2d_transpose/stack:output:0;vae/decoder/conv2d_transpose/strided_slice_1/stack:output:0=vae/decoder/conv2d_transpose/strided_slice_1/stack_1:output:0=vae/decoder/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
<vae/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpEvae_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
-vae/decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInput+vae/decoder/conv2d_transpose/stack:output:0Dvae/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0$vae/decoder/reshape/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
3vae/decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp<vae_decoder_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
$vae/decoder/conv2d_transpose/BiasAddBiasAdd6vae/decoder/conv2d_transpose/conv2d_transpose:output:0;vae/decoder/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? ?
!vae/decoder/conv2d_transpose/ReluRelu-vae/decoder/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 
 vae/decoder/decoder_output/ShapeShape/vae/decoder/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:x
.vae/decoder/decoder_output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0vae/decoder/decoder_output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0vae/decoder/decoder_output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(vae/decoder/decoder_output/strided_sliceStridedSlice)vae/decoder/decoder_output/Shape:output:07vae/decoder/decoder_output/strided_slice/stack:output:09vae/decoder/decoder_output/strided_slice/stack_1:output:09vae/decoder/decoder_output/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"vae/decoder/decoder_output/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d
"vae/decoder/decoder_output/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d
"vae/decoder/decoder_output/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
 vae/decoder/decoder_output/stackPack1vae/decoder/decoder_output/strided_slice:output:0+vae/decoder/decoder_output/stack/1:output:0+vae/decoder/decoder_output/stack/2:output:0+vae/decoder/decoder_output/stack/3:output:0*
N*
T0*
_output_shapes
:z
0vae/decoder/decoder_output/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2vae/decoder/decoder_output/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2vae/decoder/decoder_output/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*vae/decoder/decoder_output/strided_slice_1StridedSlice)vae/decoder/decoder_output/stack:output:09vae/decoder/decoder_output/strided_slice_1/stack:output:0;vae/decoder/decoder_output/strided_slice_1/stack_1:output:0;vae/decoder/decoder_output/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:vae/decoder/decoder_output/conv2d_transpose/ReadVariableOpReadVariableOpCvae_decoder_decoder_output_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
+vae/decoder/decoder_output/conv2d_transposeConv2DBackpropInput)vae/decoder/decoder_output/stack:output:0Bvae/decoder/decoder_output/conv2d_transpose/ReadVariableOp:value:0/vae/decoder/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
1vae/decoder/decoder_output/BiasAdd/ReadVariableOpReadVariableOp:vae_decoder_decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
"vae/decoder/decoder_output/BiasAddBiasAdd4vae/decoder/decoder_output/conv2d_transpose:output:09vae/decoder/decoder_output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
"vae/decoder/decoder_output/SigmoidSigmoid+vae/decoder/decoder_output/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
Avae/tf.keras.backend.binary_crossentropy/logistic_loss/zeros_like	ZerosLike+vae/decoder/decoder_output/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
Cvae/tf.keras.backend.binary_crossentropy/logistic_loss/GreaterEqualGreaterEqual+vae/decoder/decoder_output/BiasAdd:output:0Evae/tf.keras.backend.binary_crossentropy/logistic_loss/zeros_like:y:0*
T0*/
_output_shapes
:??????????
=vae/tf.keras.backend.binary_crossentropy/logistic_loss/SelectSelectGvae/tf.keras.backend.binary_crossentropy/logistic_loss/GreaterEqual:z:0+vae/decoder/decoder_output/BiasAdd:output:0Evae/tf.keras.backend.binary_crossentropy/logistic_loss/zeros_like:y:0*
T0*/
_output_shapes
:??????????
:vae/tf.keras.backend.binary_crossentropy/logistic_loss/NegNeg+vae/decoder/decoder_output/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
?vae/tf.keras.backend.binary_crossentropy/logistic_loss/Select_1SelectGvae/tf.keras.backend.binary_crossentropy/logistic_loss/GreaterEqual:z:0>vae/tf.keras.backend.binary_crossentropy/logistic_loss/Neg:y:0+vae/decoder/decoder_output/BiasAdd:output:0*
T0*/
_output_shapes
:??????????
:vae/tf.keras.backend.binary_crossentropy/logistic_loss/mulMul+vae/decoder/decoder_output/BiasAdd:output:0encoder_input*
T0*/
_output_shapes
:??????????
:vae/tf.keras.backend.binary_crossentropy/logistic_loss/subSubFvae/tf.keras.backend.binary_crossentropy/logistic_loss/Select:output:0>vae/tf.keras.backend.binary_crossentropy/logistic_loss/mul:z:0*
T0*/
_output_shapes
:??????????
:vae/tf.keras.backend.binary_crossentropy/logistic_loss/ExpExpHvae/tf.keras.backend.binary_crossentropy/logistic_loss/Select_1:output:0*
T0*/
_output_shapes
:??????????
<vae/tf.keras.backend.binary_crossentropy/logistic_loss/Log1pLog1p>vae/tf.keras.backend.binary_crossentropy/logistic_loss/Exp:y:0*
T0*/
_output_shapes
:??????????
6vae/tf.keras.backend.binary_crossentropy/logistic_lossAddV2>vae/tf.keras.backend.binary_crossentropy/logistic_loss/sub:z:0@vae/tf.keras.backend.binary_crossentropy/logistic_loss/Log1p:y:0*
T0*/
_output_shapes
:?????????j
vae/tf.math.exp/ExpExpvae/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
vae/tf.__operators__.add/AddV2AddV2
vae_115251vae/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????s
vae/tf.math.square/SquareSquarevae/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
vae/tf.math.subtract/SubSub"vae/tf.__operators__.add/AddV2:z:0vae/tf.math.square/Square:y:0*
T0*'
_output_shapes
:??????????
vae/tf.math.subtract_1/SubSubvae/tf.math.subtract/Sub:z:0vae/tf.math.exp/Exp:y:0*
T0*'
_output_shapes
:?????????y
.vae/tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
vae/tf.math.reduce_mean/MeanMeanvae/tf.math.subtract_1/Sub:z:07vae/tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*#
_output_shapes
:??????????
vae/tf.math.multiply/MulMul
vae_115259%vae/tf.math.reduce_mean/Mean:output:0*
T0*#
_output_shapes
:??????????
 vae/tf.__operators__.add_1/AddV2AddV2:vae/tf.keras.backend.binary_crossentropy/logistic_loss:z:0vae/tf.math.multiply/Mul:z:0*
T0*8
_output_shapes&
$:"??????????????????}
IdentityIdentity&vae/decoder/decoder_output/Sigmoid:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp"^vae/conv2d/BiasAdd/ReadVariableOp!^vae/conv2d/Conv2D/ReadVariableOp$^vae/conv2d_1/BiasAdd/ReadVariableOp#^vae/conv2d_1/Conv2D/ReadVariableOp$^vae/conv2d_2/BiasAdd/ReadVariableOp#^vae/conv2d_2/Conv2D/ReadVariableOp$^vae/conv2d_3/BiasAdd/ReadVariableOp#^vae/conv2d_3/Conv2D/ReadVariableOp4^vae/decoder/conv2d_transpose/BiasAdd/ReadVariableOp=^vae/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2^vae/decoder/decoder_output/BiasAdd/ReadVariableOp;^vae/decoder/decoder_output/conv2d_transpose/ReadVariableOp+^vae/decoder/dense_3/BiasAdd/ReadVariableOp*^vae/decoder/dense_3/MatMul/ReadVariableOp!^vae/dense/BiasAdd/ReadVariableOp ^vae/dense/MatMul/ReadVariableOp#^vae/dense_1/BiasAdd/ReadVariableOp"^vae/dense_1/MatMul/ReadVariableOp#^vae/dense_2/BiasAdd/ReadVariableOp"^vae/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????: : : : : : : : : : : : : : : : : : : : : : 2F
!vae/conv2d/BiasAdd/ReadVariableOp!vae/conv2d/BiasAdd/ReadVariableOp2D
 vae/conv2d/Conv2D/ReadVariableOp vae/conv2d/Conv2D/ReadVariableOp2J
#vae/conv2d_1/BiasAdd/ReadVariableOp#vae/conv2d_1/BiasAdd/ReadVariableOp2H
"vae/conv2d_1/Conv2D/ReadVariableOp"vae/conv2d_1/Conv2D/ReadVariableOp2J
#vae/conv2d_2/BiasAdd/ReadVariableOp#vae/conv2d_2/BiasAdd/ReadVariableOp2H
"vae/conv2d_2/Conv2D/ReadVariableOp"vae/conv2d_2/Conv2D/ReadVariableOp2J
#vae/conv2d_3/BiasAdd/ReadVariableOp#vae/conv2d_3/BiasAdd/ReadVariableOp2H
"vae/conv2d_3/Conv2D/ReadVariableOp"vae/conv2d_3/Conv2D/ReadVariableOp2j
3vae/decoder/conv2d_transpose/BiasAdd/ReadVariableOp3vae/decoder/conv2d_transpose/BiasAdd/ReadVariableOp2|
<vae/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp<vae/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2f
1vae/decoder/decoder_output/BiasAdd/ReadVariableOp1vae/decoder/decoder_output/BiasAdd/ReadVariableOp2x
:vae/decoder/decoder_output/conv2d_transpose/ReadVariableOp:vae/decoder/decoder_output/conv2d_transpose/ReadVariableOp2X
*vae/decoder/dense_3/BiasAdd/ReadVariableOp*vae/decoder/dense_3/BiasAdd/ReadVariableOp2V
)vae/decoder/dense_3/MatMul/ReadVariableOp)vae/decoder/dense_3/MatMul/ReadVariableOp2D
 vae/dense/BiasAdd/ReadVariableOp vae/dense/BiasAdd/ReadVariableOp2B
vae/dense/MatMul/ReadVariableOpvae/dense/MatMul/ReadVariableOp2H
"vae/dense_1/BiasAdd/ReadVariableOp"vae/dense_1/BiasAdd/ReadVariableOp2F
!vae/dense_1/MatMul/ReadVariableOp!vae/dense_1/MatMul/ReadVariableOp2H
"vae/dense_2/BiasAdd/ReadVariableOp"vae/dense_2/BiasAdd/ReadVariableOp2F
!vae/dense_2/MatMul/ReadVariableOp!vae/dense_2/MatMul/ReadVariableOp:^ Z
/
_output_shapes
:?????????
'
_user_specified_nameencoder_input:

_output_shapes
: :

_output_shapes
: 
?
?
$__inference_vae_layer_call_fn_115804
encoder_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:	?b 
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12:

unknown_13:	?b

unknown_14:	?b$

unknown_15: @

unknown_16: $

unknown_17: 

unknown_18:

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:?????????:"??????????????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_vae_layer_call_and_return_conditional_losses_115756w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????
'
_user_specified_nameencoder_input:

_output_shapes
: :

_output_shapes
: 
?
?
$__inference_signature_wrapper_116394
encoder_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:	?b 
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12:

unknown_13:	?b

unknown_14:	?b$

unknown_15: @

unknown_16: $

unknown_17: 

unknown_18:

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_115264w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????
'
_user_specified_nameencoder_input:

_output_shapes
: :

_output_shapes
: 
?
?
'__inference_conv2d_layer_call_fn_116801

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_115565w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_115302

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?*
"__inference__traced_restore_117722
file_prefix8
assignvariableop_conv2d_kernel: ,
assignvariableop_1_conv2d_bias: <
"assignvariableop_2_conv2d_1_kernel: @.
 assignvariableop_3_conv2d_1_bias:@<
"assignvariableop_4_conv2d_2_kernel:@@.
 assignvariableop_5_conv2d_2_bias:@<
"assignvariableop_6_conv2d_3_kernel:@@.
 assignvariableop_7_conv2d_3_bias:@2
assignvariableop_8_dense_kernel:	?b +
assignvariableop_9_dense_bias: 4
"assignvariableop_10_dense_1_kernel: .
 assignvariableop_11_dense_1_bias:4
"assignvariableop_12_dense_2_kernel: .
 assignvariableop_13_dense_2_bias:5
"assignvariableop_14_dense_3_kernel:	?b/
 assignvariableop_15_dense_3_bias:	?bE
+assignvariableop_16_conv2d_transpose_kernel: @7
)assignvariableop_17_conv2d_transpose_bias: C
)assignvariableop_18_decoder_output_kernel: 5
'assignvariableop_19_decoder_output_bias:'
assignvariableop_20_adam_iter:	 )
assignvariableop_21_adam_beta_1: )
assignvariableop_22_adam_beta_2: (
assignvariableop_23_adam_decay: 0
&assignvariableop_24_adam_learning_rate: #
assignvariableop_25_total: #
assignvariableop_26_count: B
(assignvariableop_27_adam_conv2d_kernel_m: 4
&assignvariableop_28_adam_conv2d_bias_m: D
*assignvariableop_29_adam_conv2d_1_kernel_m: @6
(assignvariableop_30_adam_conv2d_1_bias_m:@D
*assignvariableop_31_adam_conv2d_2_kernel_m:@@6
(assignvariableop_32_adam_conv2d_2_bias_m:@D
*assignvariableop_33_adam_conv2d_3_kernel_m:@@6
(assignvariableop_34_adam_conv2d_3_bias_m:@:
'assignvariableop_35_adam_dense_kernel_m:	?b 3
%assignvariableop_36_adam_dense_bias_m: ;
)assignvariableop_37_adam_dense_1_kernel_m: 5
'assignvariableop_38_adam_dense_1_bias_m:;
)assignvariableop_39_adam_dense_2_kernel_m: 5
'assignvariableop_40_adam_dense_2_bias_m:<
)assignvariableop_41_adam_dense_3_kernel_m:	?b6
'assignvariableop_42_adam_dense_3_bias_m:	?bL
2assignvariableop_43_adam_conv2d_transpose_kernel_m: @>
0assignvariableop_44_adam_conv2d_transpose_bias_m: J
0assignvariableop_45_adam_decoder_output_kernel_m: <
.assignvariableop_46_adam_decoder_output_bias_m:B
(assignvariableop_47_adam_conv2d_kernel_v: 4
&assignvariableop_48_adam_conv2d_bias_v: D
*assignvariableop_49_adam_conv2d_1_kernel_v: @6
(assignvariableop_50_adam_conv2d_1_bias_v:@D
*assignvariableop_51_adam_conv2d_2_kernel_v:@@6
(assignvariableop_52_adam_conv2d_2_bias_v:@D
*assignvariableop_53_adam_conv2d_3_kernel_v:@@6
(assignvariableop_54_adam_conv2d_3_bias_v:@:
'assignvariableop_55_adam_dense_kernel_v:	?b 3
%assignvariableop_56_adam_dense_bias_v: ;
)assignvariableop_57_adam_dense_1_kernel_v: 5
'assignvariableop_58_adam_dense_1_bias_v:;
)assignvariableop_59_adam_dense_2_kernel_v: 5
'assignvariableop_60_adam_dense_2_bias_v:<
)assignvariableop_61_adam_dense_3_kernel_v:	?b6
'assignvariableop_62_adam_dense_3_bias_v:	?bL
2assignvariableop_63_adam_conv2d_transpose_kernel_v: @>
0assignvariableop_64_adam_conv2d_transpose_bias_v: J
0assignvariableop_65_adam_decoder_output_kernel_v: <
.assignvariableop_66_adam_decoder_output_bias_v:
identity_68??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?#
value?#B?#DB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:D*
dtype0*?
value?B?DB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*R
dtypesH
F2D	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_3_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_3_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp+assignvariableop_16_conv2d_transpose_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp)assignvariableop_17_conv2d_transpose_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp)assignvariableop_18_decoder_output_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp'assignvariableop_19_decoder_output_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_iterIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_beta_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_beta_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_learning_rateIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_conv2d_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_conv2d_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv2d_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv2d_3_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv2d_3_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_dense_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp%assignvariableop_36_adam_dense_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_2_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_2_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_3_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_3_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp2assignvariableop_43_adam_conv2d_transpose_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp0assignvariableop_44_adam_conv2d_transpose_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp0assignvariableop_45_adam_decoder_output_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp.assignvariableop_46_adam_decoder_output_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_conv2d_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp&assignvariableop_48_adam_conv2d_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_conv2d_1_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_conv2d_1_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_conv2d_2_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_conv2d_2_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_conv2d_3_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_conv2d_3_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp'assignvariableop_55_adam_dense_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp%assignvariableop_56_adam_dense_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp)assignvariableop_57_adam_dense_1_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp'assignvariableop_58_adam_dense_1_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp)assignvariableop_59_adam_dense_2_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp'assignvariableop_60_adam_dense_2_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp)assignvariableop_61_adam_dense_3_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp'assignvariableop_62_adam_dense_3_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp2assignvariableop_63_adam_conv2d_transpose_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp0assignvariableop_64_adam_conv2d_transpose_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp0assignvariableop_65_adam_decoder_output_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp.assignvariableop_66_adam_decoder_output_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_67Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_68IdentityIdentity_67:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_68Identity_68:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
o
B__inference_lambda_layer_call_and_return_conditional_losses_115841

inputs
inputs_1
identity?;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??<?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????F
ExpExpinputs_1*
T0*'
_output_shapes
:?????????X
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????O
addAddV2inputsmul:z:0*
T0*'
_output_shapes
:?????????O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
p
D__inference_add_loss_layer_call_and_return_conditional_losses_117160

inputs
identity

identity_1_
IdentityIdentityinputs*
T0*8
_output_shapes&
$:"??????????????????a

Identity_1Identityinputs*
T0*8
_output_shapes&
$:"??????????????????"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:"??????????????????:` \
8
_output_shapes&
$:"??????????????????
 
_user_specified_nameinputs
?
_
C__inference_reshape_layer_call_and_return_conditional_losses_117199

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????@`
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????b:P L
(
_output_shapes
:??????????b
 
_user_specified_nameinputs
?

?
C__inference_dense_3_layer_call_and_return_conditional_losses_117180

inputs1
matmul_readvariableop_resource:	?b.
biasadd_readvariableop_resource:	?b
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?b*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????bs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????bQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????bb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????bw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?[
?
?__inference_vae_layer_call_and_return_conditional_losses_116248
encoder_input'
conv2d_116162: 
conv2d_116164: )
conv2d_1_116167: @
conv2d_1_116169:@)
conv2d_2_116172:@@
conv2d_2_116174:@)
conv2d_3_116177:@@
conv2d_3_116179:@
dense_116183:	?b 
dense_116185:  
dense_1_116188: 
dense_1_116190: 
dense_2_116193: 
dense_2_116195:!
decoder_116199:	?b
decoder_116201:	?b(
decoder_116203: @
decoder_116205: (
decoder_116207: 
decoder_116209:
unknown
	unknown_0
identity

identity_1??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?decoder/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?lambda/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallencoder_inputconv2d_116162conv2d_116164*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_115565?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_116167conv2d_1_116169*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_115582?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_116172conv2d_2_116174*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_115599?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_116177conv2d_3_116179*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_115616?
flatten/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_115628?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_116183dense_116185*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_115641?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_116188dense_1_116190*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_115657?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_2_116193dense_2_116195*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_115673?
lambda/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_115699?
decoder/StatefulPartitionedCallStatefulPartitionedCall'lambda/StatefulPartitionedCall:output:0decoder_116199decoder_116201decoder_116203decoder_116205decoder_116207decoder_116209*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_115405o
*tf.keras.backend.binary_crossentropy/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *???3o
*tf.keras.backend.binary_crossentropy/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
(tf.keras.backend.binary_crossentropy/subSub3tf.keras.backend.binary_crossentropy/sub/x:output:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*
_output_shapes
: ?
:tf.keras.backend.binary_crossentropy/clip_by_value/MinimumMinimum(decoder/StatefulPartitionedCall:output:0,tf.keras.backend.binary_crossentropy/sub:z:0*
T0*/
_output_shapes
:??????????
2tf.keras.backend.binary_crossentropy/clip_by_valueMaximum>tf.keras.backend.binary_crossentropy/clip_by_value/Minimum:z:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*/
_output_shapes
:?????????o
*tf.keras.backend.binary_crossentropy/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
(tf.keras.backend.binary_crossentropy/addAddV26tf.keras.backend.binary_crossentropy/clip_by_value:z:03tf.keras.backend.binary_crossentropy/add/y:output:0*
T0*/
_output_shapes
:??????????
(tf.keras.backend.binary_crossentropy/LogLog,tf.keras.backend.binary_crossentropy/add:z:0*
T0*/
_output_shapes
:??????????
(tf.keras.backend.binary_crossentropy/mulMulencoder_input,tf.keras.backend.binary_crossentropy/Log:y:0*
T0*/
_output_shapes
:?????????q
,tf.keras.backend.binary_crossentropy/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
*tf.keras.backend.binary_crossentropy/sub_1Sub5tf.keras.backend.binary_crossentropy/sub_1/x:output:0encoder_input*
T0*/
_output_shapes
:?????????q
,tf.keras.backend.binary_crossentropy/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
*tf.keras.backend.binary_crossentropy/sub_2Sub5tf.keras.backend.binary_crossentropy/sub_2/x:output:06tf.keras.backend.binary_crossentropy/clip_by_value:z:0*
T0*/
_output_shapes
:?????????q
,tf.keras.backend.binary_crossentropy/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
*tf.keras.backend.binary_crossentropy/add_1AddV2.tf.keras.backend.binary_crossentropy/sub_2:z:05tf.keras.backend.binary_crossentropy/add_1/y:output:0*
T0*/
_output_shapes
:??????????
*tf.keras.backend.binary_crossentropy/Log_1Log.tf.keras.backend.binary_crossentropy/add_1:z:0*
T0*/
_output_shapes
:??????????
*tf.keras.backend.binary_crossentropy/mul_1Mul.tf.keras.backend.binary_crossentropy/sub_1:z:0.tf.keras.backend.binary_crossentropy/Log_1:y:0*
T0*/
_output_shapes
:??????????
*tf.keras.backend.binary_crossentropy/add_2AddV2,tf.keras.backend.binary_crossentropy/mul:z:0.tf.keras.backend.binary_crossentropy/mul_1:z:0*
T0*/
_output_shapes
:??????????
(tf.keras.backend.binary_crossentropy/NegNeg.tf.keras.backend.binary_crossentropy/add_2:z:0*
T0*/
_output_shapes
:?????????r
tf.math.exp/ExpExp(dense_2/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:??????????
tf.__operators__.add/AddV2AddV2unknown(dense_2/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????{
tf.math.square/SquareSquare(dense_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:??????????
tf.math.subtract/SubSubtf.__operators__.add/AddV2:z:0tf.math.square/Square:y:0*
T0*'
_output_shapes
:?????????~
tf.math.subtract_1/SubSubtf.math.subtract/Sub:z:0tf.math.exp/Exp:y:0*
T0*'
_output_shapes
:?????????u
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
tf.math.reduce_mean/MeanMeantf.math.subtract_1/Sub:z:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*#
_output_shapes
:?????????w
tf.math.multiply/MulMul	unknown_0!tf.math.reduce_mean/Mean:output:0*
T0*#
_output_shapes
:??????????
tf.__operators__.add_1/AddV2AddV2,tf.keras.backend.binary_crossentropy/Neg:y:0tf.math.multiply/Mul:z:0*
T0*8
_output_shapes&
$:"???????????????????
add_loss/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *\
_output_shapesJ
H:"??????????????????:"??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_add_loss_layer_call_and_return_conditional_losses_115751
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????

Identity_1Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*8
_output_shapes&
$:"???????????????????
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^decoder/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^lambda/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????: : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2@
lambda/StatefulPartitionedCalllambda/StatefulPartitionedCall:^ Z
/
_output_shapes
:?????????
'
_user_specified_nameencoder_input:

_output_shapes
: :

_output_shapes
: 
?Z
?
?__inference_vae_layer_call_and_return_conditional_losses_115756

inputs'
conv2d_115566: 
conv2d_115568: )
conv2d_1_115583: @
conv2d_1_115585:@)
conv2d_2_115600:@@
conv2d_2_115602:@)
conv2d_3_115617:@@
conv2d_3_115619:@
dense_115642:	?b 
dense_115644:  
dense_1_115658: 
dense_1_115660: 
dense_2_115674: 
dense_2_115676:!
decoder_115701:	?b
decoder_115703:	?b(
decoder_115705: @
decoder_115707: (
decoder_115709: 
decoder_115711:
unknown
	unknown_0
identity

identity_1??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall? conv2d_3/StatefulPartitionedCall?decoder/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?lambda/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_115566conv2d_115568*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_115565?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_115583conv2d_1_115585*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_115582?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_115600conv2d_2_115602*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_115599?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_115617conv2d_3_115619*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_115616?
flatten/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_115628?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_115642dense_115644*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_115641?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_115658dense_1_115660*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_115657?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_2_115674dense_2_115676*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_115673?
lambda/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_115699?
decoder/StatefulPartitionedCallStatefulPartitionedCall'lambda/StatefulPartitionedCall:output:0decoder_115701decoder_115703decoder_115705decoder_115707decoder_115709decoder_115711*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_115405o
*tf.keras.backend.binary_crossentropy/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *???3o
*tf.keras.backend.binary_crossentropy/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
(tf.keras.backend.binary_crossentropy/subSub3tf.keras.backend.binary_crossentropy/sub/x:output:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*
_output_shapes
: ?
:tf.keras.backend.binary_crossentropy/clip_by_value/MinimumMinimum(decoder/StatefulPartitionedCall:output:0,tf.keras.backend.binary_crossentropy/sub:z:0*
T0*/
_output_shapes
:??????????
2tf.keras.backend.binary_crossentropy/clip_by_valueMaximum>tf.keras.backend.binary_crossentropy/clip_by_value/Minimum:z:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*/
_output_shapes
:?????????o
*tf.keras.backend.binary_crossentropy/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
(tf.keras.backend.binary_crossentropy/addAddV26tf.keras.backend.binary_crossentropy/clip_by_value:z:03tf.keras.backend.binary_crossentropy/add/y:output:0*
T0*/
_output_shapes
:??????????
(tf.keras.backend.binary_crossentropy/LogLog,tf.keras.backend.binary_crossentropy/add:z:0*
T0*/
_output_shapes
:??????????
(tf.keras.backend.binary_crossentropy/mulMulinputs,tf.keras.backend.binary_crossentropy/Log:y:0*
T0*/
_output_shapes
:?????????q
,tf.keras.backend.binary_crossentropy/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
*tf.keras.backend.binary_crossentropy/sub_1Sub5tf.keras.backend.binary_crossentropy/sub_1/x:output:0inputs*
T0*/
_output_shapes
:?????????q
,tf.keras.backend.binary_crossentropy/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
*tf.keras.backend.binary_crossentropy/sub_2Sub5tf.keras.backend.binary_crossentropy/sub_2/x:output:06tf.keras.backend.binary_crossentropy/clip_by_value:z:0*
T0*/
_output_shapes
:?????????q
,tf.keras.backend.binary_crossentropy/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
*tf.keras.backend.binary_crossentropy/add_1AddV2.tf.keras.backend.binary_crossentropy/sub_2:z:05tf.keras.backend.binary_crossentropy/add_1/y:output:0*
T0*/
_output_shapes
:??????????
*tf.keras.backend.binary_crossentropy/Log_1Log.tf.keras.backend.binary_crossentropy/add_1:z:0*
T0*/
_output_shapes
:??????????
*tf.keras.backend.binary_crossentropy/mul_1Mul.tf.keras.backend.binary_crossentropy/sub_1:z:0.tf.keras.backend.binary_crossentropy/Log_1:y:0*
T0*/
_output_shapes
:??????????
*tf.keras.backend.binary_crossentropy/add_2AddV2,tf.keras.backend.binary_crossentropy/mul:z:0.tf.keras.backend.binary_crossentropy/mul_1:z:0*
T0*/
_output_shapes
:??????????
(tf.keras.backend.binary_crossentropy/NegNeg.tf.keras.backend.binary_crossentropy/add_2:z:0*
T0*/
_output_shapes
:?????????r
tf.math.exp/ExpExp(dense_2/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:??????????
tf.__operators__.add/AddV2AddV2unknown(dense_2/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????{
tf.math.square/SquareSquare(dense_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:??????????
tf.math.subtract/SubSubtf.__operators__.add/AddV2:z:0tf.math.square/Square:y:0*
T0*'
_output_shapes
:?????????~
tf.math.subtract_1/SubSubtf.math.subtract/Sub:z:0tf.math.exp/Exp:y:0*
T0*'
_output_shapes
:?????????u
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
??????????
tf.math.reduce_mean/MeanMeantf.math.subtract_1/Sub:z:03tf.math.reduce_mean/Mean/reduction_indices:output:0*
T0*#
_output_shapes
:?????????w
tf.math.multiply/MulMul	unknown_0!tf.math.reduce_mean/Mean:output:0*
T0*#
_output_shapes
:??????????
tf.__operators__.add_1/AddV2AddV2,tf.keras.backend.binary_crossentropy/Neg:y:0tf.math.multiply/Mul:z:0*
T0*8
_output_shapes&
$:"???????????????????
add_loss/PartitionedCallPartitionedCall tf.__operators__.add_1/AddV2:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *\
_output_shapesJ
H:"??????????????????:"??????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_add_loss_layer_call_and_return_conditional_losses_115751
IdentityIdentity(decoder/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????

Identity_1Identity!add_loss/PartitionedCall:output:1^NoOp*
T0*8
_output_shapes&
$:"???????????????????
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall ^decoder/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^lambda/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????: : : : : : : : : : : : : : : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2@
lambda/StatefulPartitionedCalllambda/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_116872

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
p
D__inference_add_loss_layer_call_and_return_conditional_losses_115751

inputs
identity

identity_1_
IdentityIdentityinputs*
T0*8
_output_shapes&
$:"??????????????????a

Identity_1Identityinputs*
T0*8
_output_shapes&
$:"??????????????????"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:"??????????????????:` \
8
_output_shapes&
$:"??????????????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_3_layer_call_fn_116861

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_115616w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
A__inference_dense_layer_call_and_return_conditional_losses_115641

inputs1
matmul_readvariableop_resource:	?b -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?b *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????b: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????b
 
_user_specified_nameinputs
?
?
(__inference_dense_1_layer_call_fn_116912

inputs
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_115657o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_115582

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
(__inference_decoder_layer_call_fn_115420
decoder_input
unknown:	?b
	unknown_0:	?b#
	unknown_1: @
	unknown_2: #
	unknown_3: 
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldecoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_115405w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedecoder_input
?
q
B__inference_lambda_layer_call_and_return_conditional_losses_116993
inputs_0
inputs_1
identity?=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_normal/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
random_normal/shapePackstrided_slice:output:0random_normal/shape/1:output:0*
N*
T0*
_output_shapes
:W
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*'
_output_shapes
:?????????*
dtype0*
seed???)*
seed2??^?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????|
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:?????????F
ExpExpinputs_1*
T0*'
_output_shapes
:?????????X
mulMulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????Q
addAddV2inputs_0mul:z:0*
T0*'
_output_shapes
:?????????O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

?
A__inference_dense_layer_call_and_return_conditional_losses_116903

inputs1
matmul_readvariableop_resource:	?b -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?b *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????b: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????b
 
_user_specified_nameinputs
?	
?
C__inference_dense_2_layer_call_and_return_conditional_losses_115673

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
$__inference_vae_layer_call_fn_116444

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:	?b 
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12:

unknown_13:	?b

unknown_14:	?b$

unknown_15: @

unknown_16: $

unknown_17: 

unknown_18:

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:?????????:"??????????????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_vae_layer_call_and_return_conditional_losses_115756w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_115616

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
$__inference_vae_layer_call_fn_116159
encoder_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@
	unknown_7:	?b 
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12:

unknown_13:	?b

unknown_14:	?b$

unknown_15: @

unknown_16: $

unknown_17: 

unknown_18:

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallencoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *S
_output_shapesA
?:?????????:"??????????????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *H
fCRA
?__inference_vae_layer_call_and_return_conditional_losses_116061w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:?????????: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????
'
_user_specified_nameencoder_input:

_output_shapes
: :

_output_shapes
: 
?	
?
(__inference_decoder_layer_call_fn_115507
decoder_input
unknown:	?b
	unknown_0:	?b#
	unknown_1: @
	unknown_2: #
	unknown_3: 
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldecoder_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoder_layer_call_and_return_conditional_losses_115475w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_namedecoder_input"?	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
O
encoder_input>
serving_default_encoder_input:0?????????C
decoder8
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer_with_weights-7
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
loss

signatures"
_tf_keras_network
"
_tf_keras_input_layer
?
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses

&kernel
'bias
 (_jit_compiled_convolution_op"
_tf_keras_layer
?
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias
 1_jit_compiled_convolution_op"
_tf_keras_layer
?
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias
 :_jit_compiled_convolution_op"
_tf_keras_layer
?
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

Akernel
Bbias
 C_jit_compiled_convolution_op"
_tf_keras_layer
?
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_layer
?
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses

Pkernel
Qbias"
_tf_keras_layer
?
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
V__call__
*W&call_and_return_all_conditional_losses

Xkernel
Ybias"
_tf_keras_layer
?
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias"
_tf_keras_layer
?
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses"
_tf_keras_layer
?
hlayer-0
ilayer_with_weights-0
ilayer-1
jlayer-2
klayer_with_weights-1
klayer-3
llayer_with_weights-2
llayer-4
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_network
(
s	keras_api"
_tf_keras_layer
(
t	keras_api"
_tf_keras_layer
(
u	keras_api"
_tf_keras_layer
(
v	keras_api"
_tf_keras_layer
(
w	keras_api"
_tf_keras_layer
(
x	keras_api"
_tf_keras_layer
(
y	keras_api"
_tf_keras_layer
(
z	keras_api"
_tf_keras_layer
(
{	keras_api"
_tf_keras_layer
?
|	variables
}trainable_variables
~regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
&0
'1
/2
03
84
95
A6
B7
P8
Q9
X10
Y11
`12
a13
?14
?15
?16
?17
?18
?19"
trackable_list_wrapper
?
&0
'1
/2
03
84
95
A6
B7
P8
Q9
X10
Y11
`12
a13
?14
?15
?16
?17
?18
?19"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
$__inference_vae_layer_call_fn_115804
$__inference_vae_layer_call_fn_116444
$__inference_vae_layer_call_fn_116494
$__inference_vae_layer_call_fn_116159?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
?__inference_vae_layer_call_and_return_conditional_losses_116643
?__inference_vae_layer_call_and_return_conditional_losses_116792
?__inference_vae_layer_call_and_return_conditional_losses_116248
?__inference_vae_layer_call_and_return_conditional_losses_116337?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?
capture_20
?
capture_21B?
!__inference__wrapped_model_115264encoder_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?
capture_20z?
capture_21
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate&m?'m?/m?0m?8m?9m?Am?Bm?Pm?Qm?Xm?Ym?`m?am?	?m?	?m?	?m?	?m?	?m?	?m?&v?'v?/v?0v?8v?9v?Av?Bv?Pv?Qv?Xv?Yv?`v?av?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
 "
trackable_dict_wrapper
-
?serving_default"
signature_map
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
'__inference_conv2d_layer_call_fn_116801?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
B__inference_conv2d_layer_call_and_return_conditional_losses_116812?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
':% 2conv2d/kernel
: 2conv2d/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_conv2d_1_layer_call_fn_116821?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_116832?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
):' @2conv2d_1/kernel
:@2conv2d_1/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_conv2d_2_layer_call_fn_116841?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_116852?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
):'@@2conv2d_2/kernel
:@2conv2d_2/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_conv2d_3_layer_call_fn_116861?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_116872?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
):'@@2conv2d_3/kernel
:@2conv2d_3/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_flatten_layer_call_fn_116877?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_flatten_layer_call_and_return_conditional_losses_116883?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
&__inference_dense_layer_call_fn_116892?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
A__inference_dense_layer_call_and_return_conditional_losses_116903?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
:	?b 2dense/kernel
: 2
dense/bias
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
R	variables
Strainable_variables
Tregularization_losses
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_dense_1_layer_call_fn_116912?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_dense_1_layer_call_and_return_conditional_losses_116922?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 : 2dense_1/kernel
:2dense_1/bias
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_dense_2_layer_call_fn_116931?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_dense_2_layer_call_and_return_conditional_losses_116941?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 : 2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
'__inference_lambda_layer_call_fn_116947
'__inference_lambda_layer_call_fn_116953?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
B__inference_lambda_layer_call_and_return_conditional_losses_116973
B__inference_lambda_layer_call_and_return_conditional_losses_116993?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
"
_tf_keras_input_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
!?_jit_compiled_convolution_op"
_tf_keras_layer
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
P
?0
?1
?2
?3
?4
?5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
(__inference_decoder_layer_call_fn_115420
(__inference_decoder_layer_call_fn_117010
(__inference_decoder_layer_call_fn_117027
(__inference_decoder_layer_call_fn_115507?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
C__inference_decoder_layer_call_and_return_conditional_losses_117088
C__inference_decoder_layer_call_and_return_conditional_losses_117149
C__inference_decoder_layer_call_and_return_conditional_losses_115527
C__inference_decoder_layer_call_and_return_conditional_losses_115547?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
|	variables
}trainable_variables
~regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_add_loss_layer_call_fn_117155?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_add_loss_layer_call_and_return_conditional_losses_117160?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
!:	?b2dense_3/kernel
:?b2dense_3/bias
1:/ @2conv2d_transpose/kernel
#:! 2conv2d_transpose/bias
/:- 2decoder_output/kernel
!:2decoder_output/bias
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
?
capture_20
?
capture_21B?
$__inference_vae_layer_call_fn_115804encoder_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?
capture_20z?
capture_21
?
?
capture_20
?
capture_21B?
$__inference_vae_layer_call_fn_116444inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?
capture_20z?
capture_21
?
?
capture_20
?
capture_21B?
$__inference_vae_layer_call_fn_116494inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?
capture_20z?
capture_21
?
?
capture_20
?
capture_21B?
$__inference_vae_layer_call_fn_116159encoder_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?
capture_20z?
capture_21
?
?
capture_20
?
capture_21B?
?__inference_vae_layer_call_and_return_conditional_losses_116643inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?
capture_20z?
capture_21
?
?
capture_20
?
capture_21B?
?__inference_vae_layer_call_and_return_conditional_losses_116792inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?
capture_20z?
capture_21
?
?
capture_20
?
capture_21B?
?__inference_vae_layer_call_and_return_conditional_losses_116248encoder_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?
capture_20z?
capture_21
?
?
capture_20
?
capture_21B?
?__inference_vae_layer_call_and_return_conditional_losses_116337encoder_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?
capture_20z?
capture_21
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?
?
capture_20
?
capture_21B?
$__inference_signature_wrapper_116394encoder_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?
capture_20z?
capture_21
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_conv2d_layer_call_fn_116801inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_conv2d_layer_call_and_return_conditional_losses_116812inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_conv2d_1_layer_call_fn_116821inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_116832inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_conv2d_2_layer_call_fn_116841inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_116852inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_conv2d_3_layer_call_fn_116861inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_116872inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_flatten_layer_call_fn_116877inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_flatten_layer_call_and_return_conditional_losses_116883inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
&__inference_dense_layer_call_fn_116892inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
A__inference_dense_layer_call_and_return_conditional_losses_116903inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_dense_1_layer_call_fn_116912inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_dense_1_layer_call_and_return_conditional_losses_116922inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_dense_2_layer_call_fn_116931inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_dense_2_layer_call_and_return_conditional_losses_116941inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_lambda_layer_call_fn_116947inputs/0inputs/1"?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_lambda_layer_call_fn_116953inputs/0inputs/1"?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_lambda_layer_call_and_return_conditional_losses_116973inputs/0inputs/1"?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
B__inference_lambda_layer_call_and_return_conditional_losses_116993inputs/0inputs/1"?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_dense_3_layer_call_fn_117169?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_dense_3_layer_call_and_return_conditional_losses_117180?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
(__inference_reshape_layer_call_fn_117185?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
C__inference_reshape_layer_call_and_return_conditional_losses_117199?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
1__inference_conv2d_transpose_layer_call_fn_117208?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_117242?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_decoder_output_layer_call_fn_117251?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
J__inference_decoder_output_layer_call_and_return_conditional_losses_117285?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
C
h0
i1
j2
k3
l4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_decoder_layer_call_fn_115420decoder_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
(__inference_decoder_layer_call_fn_117010inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
(__inference_decoder_layer_call_fn_117027inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
(__inference_decoder_layer_call_fn_115507decoder_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_decoder_layer_call_and_return_conditional_losses_117088inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_decoder_layer_call_and_return_conditional_losses_117149inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_decoder_layer_call_and_return_conditional_losses_115527decoder_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_decoder_layer_call_and_return_conditional_losses_115547decoder_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
)__inference_add_loss_layer_call_fn_117155inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_add_loss_layer_call_and_return_conditional_losses_117160inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_dense_3_layer_call_fn_117169inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_dense_3_layer_call_and_return_conditional_losses_117180inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
(__inference_reshape_layer_call_fn_117185inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
C__inference_reshape_layer_call_and_return_conditional_losses_117199inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
1__inference_conv2d_transpose_layer_call_fn_117208inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_117242inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
/__inference_decoder_output_layer_call_fn_117251inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_decoder_output_layer_call_and_return_conditional_losses_117285inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
,:* 2Adam/conv2d/kernel/m
: 2Adam/conv2d/bias/m
.:, @2Adam/conv2d_1/kernel/m
 :@2Adam/conv2d_1/bias/m
.:,@@2Adam/conv2d_2/kernel/m
 :@2Adam/conv2d_2/bias/m
.:,@@2Adam/conv2d_3/kernel/m
 :@2Adam/conv2d_3/bias/m
$:"	?b 2Adam/dense/kernel/m
: 2Adam/dense/bias/m
%:# 2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
%:# 2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
&:$	?b2Adam/dense_3/kernel/m
 :?b2Adam/dense_3/bias/m
6:4 @2Adam/conv2d_transpose/kernel/m
(:& 2Adam/conv2d_transpose/bias/m
4:2 2Adam/decoder_output/kernel/m
&:$2Adam/decoder_output/bias/m
,:* 2Adam/conv2d/kernel/v
: 2Adam/conv2d/bias/v
.:, @2Adam/conv2d_1/kernel/v
 :@2Adam/conv2d_1/bias/v
.:,@@2Adam/conv2d_2/kernel/v
 :@2Adam/conv2d_2/bias/v
.:,@@2Adam/conv2d_3/kernel/v
 :@2Adam/conv2d_3/bias/v
$:"	?b 2Adam/dense/kernel/v
: 2Adam/dense/bias/v
%:# 2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
%:# 2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
&:$	?b2Adam/dense_3/kernel/v
 :?b2Adam/dense_3/bias/v
6:4 @2Adam/conv2d_transpose/kernel/v
(:& 2Adam/conv2d_transpose/bias/v
4:2 2Adam/decoder_output/kernel/v
&:$2Adam/decoder_output/bias/v?
!__inference__wrapped_model_115264?&'/089ABPQXY`a????????>?;
4?1
/?,
encoder_input?????????
? "9?6
4
decoder)?&
decoder??????????
D__inference_add_loss_layer_call_and_return_conditional_losses_117160?@?=
6?3
1?.
inputs"??????????????????
? "f?c
,?)
0"??????????????????
3?0
.?+
1/0"???????????????????
)__inference_add_loss_layer_call_fn_117155m@?=
6?3
1?.
inputs"??????????????????
? ")?&"???????????????????
D__inference_conv2d_1_layer_call_and_return_conditional_losses_116832l/07?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
)__inference_conv2d_1_layer_call_fn_116821_/07?4
-?*
(?%
inputs????????? 
? " ??????????@?
D__inference_conv2d_2_layer_call_and_return_conditional_losses_116852l897?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
)__inference_conv2d_2_layer_call_fn_116841_897?4
-?*
(?%
inputs?????????@
? " ??????????@?
D__inference_conv2d_3_layer_call_and_return_conditional_losses_116872lAB7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
)__inference_conv2d_3_layer_call_fn_116861_AB7?4
-?*
(?%
inputs?????????@
? " ??????????@?
B__inference_conv2d_layer_call_and_return_conditional_losses_116812l&'7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
'__inference_conv2d_layer_call_fn_116801_&'7?4
-?*
(?%
inputs?????????
? " ?????????? ?
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_117242???I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
1__inference_conv2d_transpose_layer_call_fn_117208???I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
C__inference_decoder_layer_call_and_return_conditional_losses_115527}??????>?;
4?1
'?$
decoder_input?????????
p 

 
? "-?*
#? 
0?????????
? ?
C__inference_decoder_layer_call_and_return_conditional_losses_115547}??????>?;
4?1
'?$
decoder_input?????????
p

 
? "-?*
#? 
0?????????
? ?
C__inference_decoder_layer_call_and_return_conditional_losses_117088v??????7?4
-?*
 ?
inputs?????????
p 

 
? "-?*
#? 
0?????????
? ?
C__inference_decoder_layer_call_and_return_conditional_losses_117149v??????7?4
-?*
 ?
inputs?????????
p

 
? "-?*
#? 
0?????????
? ?
(__inference_decoder_layer_call_fn_115420p??????>?;
4?1
'?$
decoder_input?????????
p 

 
? " ???????????
(__inference_decoder_layer_call_fn_115507p??????>?;
4?1
'?$
decoder_input?????????
p

 
? " ???????????
(__inference_decoder_layer_call_fn_117010i??????7?4
-?*
 ?
inputs?????????
p 

 
? " ???????????
(__inference_decoder_layer_call_fn_117027i??????7?4
-?*
 ?
inputs?????????
p

 
? " ???????????
J__inference_decoder_output_layer_call_and_return_conditional_losses_117285???I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
/__inference_decoder_output_layer_call_fn_117251???I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
C__inference_dense_1_layer_call_and_return_conditional_losses_116922\XY/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? {
(__inference_dense_1_layer_call_fn_116912OXY/?,
%?"
 ?
inputs????????? 
? "???????????
C__inference_dense_2_layer_call_and_return_conditional_losses_116941\`a/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? {
(__inference_dense_2_layer_call_fn_116931O`a/?,
%?"
 ?
inputs????????? 
? "???????????
C__inference_dense_3_layer_call_and_return_conditional_losses_117180_??/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????b
? ~
(__inference_dense_3_layer_call_fn_117169R??/?,
%?"
 ?
inputs?????????
? "???????????b?
A__inference_dense_layer_call_and_return_conditional_losses_116903]PQ0?-
&?#
!?
inputs??????????b
? "%?"
?
0????????? 
? z
&__inference_dense_layer_call_fn_116892PPQ0?-
&?#
!?
inputs??????????b
? "?????????? ?
C__inference_flatten_layer_call_and_return_conditional_losses_116883a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????b
? ?
(__inference_flatten_layer_call_fn_116877T7?4
-?*
(?%
inputs?????????@
? "???????????b?
B__inference_lambda_layer_call_and_return_conditional_losses_116973?b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????

 
p 
? "%?"
?
0?????????
? ?
B__inference_lambda_layer_call_and_return_conditional_losses_116993?b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????

 
p
? "%?"
?
0?????????
? ?
'__inference_lambda_layer_call_fn_116947~b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????

 
p 
? "???????????
'__inference_lambda_layer_call_fn_116953~b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????

 
p
? "???????????
C__inference_reshape_layer_call_and_return_conditional_losses_117199a0?-
&?#
!?
inputs??????????b
? "-?*
#? 
0?????????@
? ?
(__inference_reshape_layer_call_fn_117185T0?-
&?#
!?
inputs??????????b
? " ??????????@?
$__inference_signature_wrapper_116394?&'/089ABPQXY`a????????O?L
? 
E?B
@
encoder_input/?,
encoder_input?????????"9?6
4
decoder)?&
decoder??????????
?__inference_vae_layer_call_and_return_conditional_losses_116248?&'/089ABPQXY`a????????F?C
<?9
/?,
encoder_input?????????
p 

 
? "]?Z
#? 
0?????????
3?0
.?+
1/0"???????????????????
?__inference_vae_layer_call_and_return_conditional_losses_116337?&'/089ABPQXY`a????????F?C
<?9
/?,
encoder_input?????????
p

 
? "]?Z
#? 
0?????????
3?0
.?+
1/0"???????????????????
?__inference_vae_layer_call_and_return_conditional_losses_116643?&'/089ABPQXY`a??????????<
5?2
(?%
inputs?????????
p 

 
? "]?Z
#? 
0?????????
3?0
.?+
1/0"???????????????????
?__inference_vae_layer_call_and_return_conditional_losses_116792?&'/089ABPQXY`a??????????<
5?2
(?%
inputs?????????
p

 
? "]?Z
#? 
0?????????
3?0
.?+
1/0"???????????????????
$__inference_vae_layer_call_fn_115804?&'/089ABPQXY`a????????F?C
<?9
/?,
encoder_input?????????
p 

 
? " ???????????
$__inference_vae_layer_call_fn_116159?&'/089ABPQXY`a????????F?C
<?9
/?,
encoder_input?????????
p

 
? " ???????????
$__inference_vae_layer_call_fn_116444?&'/089ABPQXY`a??????????<
5?2
(?%
inputs?????????
p 

 
? " ???????????
$__inference_vae_layer_call_fn_116494?&'/089ABPQXY`a??????????<
5?2
(?%
inputs?????????
p

 
? " ??????????