module Utils where

import           Torch
import qualified Torch.Internal.Managed.Native as ATen
import           Torch.Internal.Cast
import           System.IO.Unsafe

import           Types

  
gramMatrix t = features `bmm` featuresT / asTensor (ch * w * h)
  where [b, ch, w, h] = shape t
        features = reshape [b, ch, w * h] t
        featuresT = transpose (Dim 1) (Dim 2) features 
  
upsample_nearest2d
  :: Tensor -- ^ self
  -> [Int] -- ^ output_size
  -> Double -- ^ scales_h
  -> Double -- ^ scales_w
  -> Tensor
upsample_nearest2d _self _output_size _scales_h _scales_w =
  unsafePerformIO $ (cast4 ATen.upsample_nearest2d_tldd) _self
  _output_size _scales_h _scales_w

upsample_bilinear2d
  :: Tensor -- ^ self
  -> [Int]  -- ^ output_size
  -> Bool -- ^ align_corners
  -> Double -- ^ scales_h
  -> Double -- ^ scales_w
  -> Tensor
upsample_bilinear2d _self _output_size _align_corners _scales_h _scales_w = unsafePerformIO $ (cast5 ATen.upsample_bilinear2d_tlbdd) _self _output_size _align_corners _scales_h _scales_w
convLayer kernelSize conv =
  -- flip reflection_pad2d (padding,padding,padding,padding) . conv2dForward conv (2, 2) (1,1)
  -- flip reflection_pad1d (padding,padding) . conv2dForward conv (2, 2) (1,1)
   conv2dForward conv (2, 2) (1,1)
  where padding = kernelSize `div` 2

residualBlock ResidualBlock{..} t =
  (relu  . convLayer 3 block1) t +  (relu . convLayer 3 block2) t

upsample size scale1 scale2 tensor =  upsample_bilinear2d tensor size True scale1 scale2 

upsampleConvLayer kernelSize scale conv =
  -- conv2dForward (2, 2) (1,1) . upsample_nearest2d 0 scale . flip reflection_pad2d (pad, pad, pad, pad)  
  -- conv2dForward conv (2, 2) (1,1) . flip reflection_pad2d (pad, pad, pad, pad)
  upsample [512,512] scale scale . conv2dForward conv (2, 2) (1,1) 
  where pad =  kernelSize `div` 2

conv2dRelu conv stride padding = relu . conv2dForward conv stride padding

vgg v@Vgg16{..} str pad =
  linear l3 .
  linear l2 . 
  linear l1 .
  flatten (Dim 1) (Dim (-1)) . 
  adaptiveAvgPool2d (7,7) .
  maxPool2d (2,2) (2,2) (0,0) (1,1) False .
  conv2dRelu c13 str pad .
  slice4' v .
  slice3' v .
  slice2' v .
  slice1' v 
vggNoFinal v@Vgg16{..} str pad =
  linear l2 . 
  linear l1  .
  flatten (Dim 1) (Dim (-1)) . 
  adaptiveAvgPool2d (7,7) .
  maxPool2d (2,2) (2,2) (0,0) (1,1) False  .
  conv2dRelu c13 str pad .
  slice4' v .
  slice3' v .
  slice2' v .
  slice1' v 
  

-- https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/vgg.py
slice4 Vgg16{..} str pad= 
  maxPool2d (2,2) (2,2) (0,0) (1,1) False . 
  conv2dRelu c12 str pad .
  conv2dRelu c11 str pad .
  conv2dRelu c10 str pad 

slice3 Vgg16{..} str pad=
  conv2dRelu c9 str pad .
  maxPool2d (2,2) (2,2) (0,0) (1,1) False . 
  conv2dRelu c8 str pad .
  conv2dRelu c7 str pad .
  conv2dRelu c6 str pad 

slice2 Vgg16{..} str pad=
  conv2dRelu c5 str pad .
  maxPool2d (2,2) (2,2) (0,0) (1,1) False . 
  conv2dRelu c4 str pad .
  conv2dRelu c3 str pad .
  maxPool2d (2,2) (2,2) (0,0) (1,1) False  

slice1 Vgg16{..} str pad=
  conv2dRelu c2 str pad .
  conv2dRelu c1 str pad
  
slice4' vgg = slice4 vgg (1,1) (1,1)
slice3' vgg = slice3 vgg (1,1) (1,1)
slice2' vgg = slice2 vgg (1,1) (1,1)
slice1' vgg = slice1 vgg (1,1) (1,1)
