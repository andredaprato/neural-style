module Main where

import           Torch.Typed.Serialize
import           Torch.DType as D
import           Torch
import           Torch.Functional.Internal (reflection_pad2d, reflection_pad1d)

import           Torch.Serialize as S
import           Torch.NN (Randomizable(sample))
import qualified Torch.Device as D
import           Torch.HList (hmapM', HList, HMap'(hmap'))
import           GHC.Generics (Generic)
import           Torch.Vision

import           System.Directory
import qualified Streamly.Prelude as S
import qualified Streamly as S
import qualified Codec.Picture as I

import           Types
import           Utils
import Mnist


transformer TransformerNet{..} =
  convLayer 9 deconv3 .  
  upsampleWithInstance 3 1 deconv2 .
  upsampleWithInstance 3 1 deconv1 .
  residualBlock res5 .
  residualBlock res4 .
  residualBlock res3 .
  residualBlock res2 .
  residualBlock res1 .
  convWithInstance trans3 .
  convLayer 3 trans2 .
  convLayer 3 trans1 
  where convWithInstance conv = relu . convLayer 2 conv
        upsampleWithInstance kernelSize scale conv = relu . upsampleConvLayer kernelSize scale conv
  
  -- add dropout
-- vgg v@Vgg16{..} str pad =
--   linear l3 .
--   linear l2 . 
--   linear l1 .
--   flatten (Dim 1) (Dim (-1)) . 
--   adaptiveAvgPool2d (7,7) .
--   maxPool2d (2,2) (2,2) (0,0) (1,1) False .
--   conv2dRelu c13 str pad .
--   slice4' v .
--   slice3' v .
--   slice2' v .
--   slice1' v 


train style vggParams tfParams loss x contentWeight styleWeight = contentLoss + styleLoss
  where y = transformer tfParams x
        featuresY = vgg vggParams (1,1) (1,1) y
        featuresX = vgg vggParams (1,1) (1,1) x 
        gramY = gramSlice vggParams (1,1) (1,1) y
        contentLoss = asTensor contentWeight * (slice2 vggParams (1,1) (1,1) y `loss` slice2 vggParams (1,1) (1,1) x)
        styleWeights = asTensor styleWeight
        styleLoss = (*) styleWeights . sum $ loss <$> gramY <*> gramStyle vggParams style
        
gramStyle params style = gramSlice params (1,1) (1,1) style

gramSlice ::  Vgg16 -> (Int,Int) -> (Int,Int) -> Tensor -> [Tensor]
gramSlice  vgg str pad t = fmap gramMatrix $  [slice1  , slice2 , slice3, slice4] <*> pure vgg <*> pure str <*> pure pad <*> pure t 
-- gramSlice  vgg str pad t = fmap gramMatrix $  (t . str . pad . vgg) <$> [slice1  , slice2 , slice3, slice4] 
  
mkAdam lr = Adam {
  beta1 = 0.9,
  beta2 = 0.999,
  m1 = [zeros' [1], zeros' [1]],
  m2 = [zeros' [1], zeros' [1]],
  iter = 0
  }

imageTransform = undefined

main :: IO ()
main = do

  mnistMain
  -- model <- sample vggSpec 
  -- let nums = ones' [1,3,224,224]
  -- vggParams <- S.load "build/vgg16.pt" 
  -- bbparams <- traverse makeIndependent $ vggParams
  -- let pretrainedbb = replaceParameters model bbparams

  -- let out = (conv2dForward (c1 pretrainedbb) (1,1) (1,1) nums) @@ (0 :: Int, 0 :: Int, 0 :: Int, 50 :: Int)
  -- print out
  -- transformer <- sample transformerSpec 
  -- contents <-  listDirectory "val2017"
  
  -- content <- randnIO' [1,3,224,224]
  -- style <-  randnIO' [1,3,224,224]

  -- either <- readImage "imagenette2-160/train/n02102040/ILSVRC2012_val_00001968.JPEG"
  -- -- let classifier =  vgg weights (1,1) (1,1)
  -- -- let random = train style vggParams transformer mseLoss content (0.5 :: Double) (0.5 :: Double) 
  -- -- print random 

  -- case either of
  --   Left err -> print err
  --   Right (img, format) -> do
  --     let resized =  upsample_nearest2d (toType Float $ hwc2chw img) [224,224] 0 0
  --     writeImage 224 224 3 (I.PixelYCbCr8 0 0 0 ) (toType UInt8 resized) >>= I.saveJpgImage 100 "img.jpg" . I.ImageYCbCr8
 --     pure ()
