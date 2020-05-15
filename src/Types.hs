{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}
module Types where

import Torch
import GHC.Generics (Generic)

type ConvLayer = Conv2d

type UpsampleConvLayer = Conv2d
data ResidualBlock = ResidualBlock
  {
  block1 :: Conv2d,
  block2 :: Conv2d
  } deriving (Generic, Show)

data ResidualBlockSpec = ResidualBlockSpec {
  channels :: Int,
  kernelSize :: Int
  -- blockSpec2 :: Conv2dSpec
  }

instance Randomizable ResidualBlockSpec ResidualBlock where
  sample ResidualBlockSpec{..} =
    ResidualBlock
    <$> sample (Conv2dSpec channels channels 3 3)
    <*> sample (Conv2dSpec channels channels kernelSize kernelSize)

data TransformerNetSpec = TransformerNetSpec {
  transSpec1 :: Conv2dSpec,
  transSpec2 :: Conv2dSpec,
  transSpec3 :: Conv2dSpec
  resBlock1  :: ResidualBlockSpec,
  resBlock2  :: ResidualBlockSpec,
  resBlock3  :: ResidualBlockSpec,
  resBlock4  :: ResidualBlockSpec,
  resBlock5  :: ResidualBlockSpec
  deconvSpec1 :: Conv2dSpec,
  deconvSpec2 :: Conv2dSpec,
  deconvSpec3 :: Conv2dSpec
  }
data TransformerNet = TransformerNet
  {
    trans1  :: Conv2d,
    trans2  :: Conv2d,
    trans3  :: Conv2d
    res1    :: ResidualBlock,
    res2    :: ResidualBlock,
    res3    :: ResidualBlock,
    res4    :: ResidualBlock,
    res5    :: ResidualBlock
    deconv1 :: UpsampleConvLayer,
    deconv2 :: UpsampleConvLayer,
    deconv3    :: Conv2d
  } deriving (Generic, Show)

instance  Randomizable TransformerNetSpec TransformerNet where
  sample TransformerNetSpec{..} =
    TransformerNet
    <$> sample transSpec1
    <*> sample transSpec2
    <*> sample transSpec3
    <*> sample resBlock1
    <*> sample resBlock2
    <*> sample resBlock3
    <*> sample resBlock4
    <*> sample resBlock5
    <*> sample deconvSpec1
    <*> sample deconvSpec2
    <*> sample deconvSpec3

data VggSpec = VggSpec {
       conv1  :: Conv2dSpec ,
       conv2  :: Conv2dSpec ,
       conv3  :: Conv2dSpec ,
       conv4  :: Conv2dSpec ,
       conv5  :: Conv2dSpec ,
       conv6  :: Conv2dSpec ,
       conv7  :: Conv2dSpec ,
       conv8  :: Conv2dSpec ,
       conv9  :: Conv2dSpec ,
       conv10  :: Conv2dSpec ,
       conv11  :: Conv2dSpec ,
       conv12  :: Conv2dSpec ,
       conv13  :: Conv2dSpec ,
       linear1:: LinearSpec ,
       linear2:: LinearSpec ,
       linear3:: LinearSpec 
  } deriving (Generic)

data Vgg16 = Vgg16 {
 c1 ::  Conv2d,
 c2 ::  Conv2d,
 c3 ::  Conv2d,
 c4 ::  Conv2d,
 c5 ::  Conv2d,
 c6 ::  Conv2d,
 c7 ::  Conv2d,
 c8 ::  Conv2d,
 c9 ::  Conv2d,
 c10 ::  Conv2d,
 c11 ::  Conv2d,
 c12 ::  Conv2d,
 c13 ::  Conv2d,
 l1 :: Linear,
 l2 :: Linear,
 l3 :: Linear
 } deriving (Show, Generic, Parameterized)

instance Randomizable VggSpec Vgg16 where
  sample VggSpec{..} =
    Vgg16
    <$> sample conv1
    <*> sample conv2
    <*> sample conv3
    <*> sample conv4
    <*> sample conv5
    <*> sample conv6
    <*> sample conv7
    <*> sample conv8
    <*> sample conv9
    <*> sample conv10
    <*> sample conv11
    <*> sample conv12
    <*> sample conv13
    <*> sample linear1
    <*> sample linear2
    <*> sample linear3
    
-- residualBlockSpec channels kernelSize = ResidualBlockSpec channels kernelSize
transformerSpec = 
  TransformerNetSpec
  (Conv2dSpec 3 32 9 9) 
  (Conv2dSpec 32 64 3 3) 
  (Conv2dSpec 64 128 3 3) 
  (ResidualBlockSpec 128 3) 
  (ResidualBlockSpec 128 3) 
  (ResidualBlockSpec 128 3) 
  (ResidualBlockSpec 128 3) 
  (ResidualBlockSpec 128 3) 
  (Conv2dSpec 128 64 3 3)
  (Conv2dSpec 64 32 3 3)
  (Conv2dSpec 32 3 3 3)

vggSpec = VggSpec
        (Conv2dSpec 3 64 3 3 ) 
        (Conv2dSpec 64 64 3 3 ) 
        (Conv2dSpec 64 128 3 3 ) 
        (Conv2dSpec 128 128 3 3 ) 
        (Conv2dSpec 128 256 3 3 ) 
        (Conv2dSpec 256 256 3 3 ) 
        (Conv2dSpec 256 256 3 3 ) 
        (Conv2dSpec 256 256 3 3 ) 
        (Conv2dSpec 256 512 3 3 ) 
        (Conv2dSpec 512 512 3 3 ) 
        (Conv2dSpec 512 512 3 3 ) 
        (Conv2dSpec 512 512 3 3 ) 
        (Conv2dSpec 512 512 3 3 ) 
        (LinearSpec (512 * 7 * 7) 4096) 
        (LinearSpec 4096 4096) 
        (LinearSpec 4096 1000)
