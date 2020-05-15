{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE UndecidableInstances #-}
module Vgg where

import Torch.Typed
import Torch.Script
import Torch.Typed.NN (Dropout, Conv2d, Conv2dSpec(..), Linear, LinearSpec(..), linear, conv2d)
import Torch.DType as D
import           GHC.Generics
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import Torch.Typed.Functional (maxPool2d, relu, dropout)
import Torch.Typed.Parameter (Parameterized)
import Torch.NN (Randomizable)
import qualified Torch.Device as D
import qualified Torch as A
import Torch.Typed.Tensor (KnownDevice, KnownDType)
import Torch.Typed.Factories (RandDTypeIsValid)
-- questions

-- what is device for? distributed training?

-- plan to implement neural style from pytorch examples with the typed api

--https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

type KernelSize = '(3,3)
type Stride = '(2,2)
type Padding = '(1,1)

data VggSpec (dtype :: D.DType) (device :: (D.DeviceType, Nat))
  = VggSpec deriving (Show, Eq)

  
data Vgg (dtype :: D.DType) device = Vgg {
  c8 :: Conv2d 64 128 3 3 dtype device,
  c7 :: Conv2d 128 64 3 3 dtype device,
  c6 :: Conv2d 256 128 3 3 dtype device,
  c5 :: Conv2d 256 256 3 3 dtype device,
  c4 :: Conv2d 512 256 3 3 dtype device,
  c3 :: Conv2d 512 512 3 3 dtype device,
  c2 :: Conv2d 512 512 3 3 dtype device,
  c1 :: Conv2d 512 512 3 3 dtype device,
  l1 :: Linear (512*7*7) 4096 dtype device,
  -- d1 :: Dropout,
  l2 :: Linear 4096 4096 dtype device,
  -- d2 :: Dropout,
  l3 :: Linear 4096 4096 dtype device
  -- d3 :: Dropout
  } deriving (Show, Generic)

instance ( KnownDType dtype
         , KnownDevice device
         , RandDTypeIsValid device dtype
         )
          => Randomizable (VggSpec dtype device) (Vgg dtype device) where
  sample VggSpec =
    Vgg
    <$> A.sample (Conv2dSpec @64 @128 @3 @3)
    <*> A.sample (Conv2dSpec @128 @64  @3 @3)
    <*> A.sample (Conv2dSpec @256 @128 @3 @3)
    <*> A.sample (Conv2dSpec @256 @256 @3 @3)
    <*> A.sample (Conv2dSpec @512 @256 @3 @3)
    <*> A.sample (Conv2dSpec @512 @512 @3 @3)
    <*> A.sample (Conv2dSpec @512 @512 @3 @3)
    <*> A.sample (Conv2dSpec @512 @512 @3 @3)
    <*> A.sample (LinearSpec @(512*7*7) @4096)
    <*> A.sample (LinearSpec @4096 @4096 )
    <*> A.sample (LinearSpec @4096 @4096 )
                
                
  

