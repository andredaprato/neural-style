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

convLayer kernelSize conv =
  -- flip reflection_pad2d (padding,padding,padding,padding) . conv2dForward conv (2, 2) (1,1)
  -- flip reflection_pad1d (padding,padding) . conv2dForward conv (2, 2) (1,1)
   conv2dForward conv (2, 2) (1,1)
  where padding = kernelSize `div` 2

residualBlock ResidualBlock{..} t =
  (relu  . convLayer 3 block1) t +  (relu . convLayer 3 block2) t

upsample size scale1 scale2 tensor =  upsample_nearest2d tensor size scale1 scale2 

upsampleConvLayer kernelSize scale conv =
  -- conv2dForward (2, 2) (1,1) . upsample_nearest2d 0 scale . flip reflection_pad2d (pad, pad, pad, pad)  
  -- conv2dForward conv (2, 2) (1,1) . flip reflection_pad2d (pad, pad, pad, pad)
  upsample [512,512] scale scale . conv2dForward conv (2, 2) (1,1) 
  where pad =  kernelSize `div` 2

conv2dRelu conv stride padding = relu . conv2dForward conv stride padding
