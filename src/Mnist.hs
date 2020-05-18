module Mnist where

import Torch
import Control.Monad (when)
import System.Random (mkStdGen, randoms)

import Torch.Functional as F
import Types
import Utils
import qualified Torch.Vision as V
import qualified Torch.Typed.Vision as V hiding (getImages')
import           Torch.Serialize as S
import GHC.Generics

import qualified Pipes as P
import qualified Pipes.Concurrent as P

randomIndexes :: Int -> [Int]
randomIndexes size = (`mod` size) <$> randoms seed where seed = mkStdGen 123

data MLPSpec = MLPSpec {
    inputFeatures :: Int,
    hiddenFeatures0 :: Int,
    hiddenFeatures1 :: Int,
    outputFeatures :: Int
    } deriving (Show, Eq)

data MLP = MLP { 
    l0 :: Linear,
    l1 :: Linear,
    l2 :: Linear
    } deriving (Generic, Show)

instance Parameterized MLP
instance Randomizable MLPSpec MLP where
    sample MLPSpec {..} = MLP 
        <$> sample (LinearSpec inputFeatures hiddenFeatures0)
        <*> sample (LinearSpec hiddenFeatures0 hiddenFeatures1)
        <*> sample (LinearSpec hiddenFeatures1 outputFeatures)


mlp :: MLP -> Tensor -> Tensor
mlp MLP{..} input = 
    logSoftmax (Dim 1)
    . linear l2
    . relu
    . linear l1
    . relu
    . linear l0
    $ input

  
normalize :: Int -> Tensor -> Tensor
normalize batchSize img = (img / asTensor (255 :: Float) - mean) / std 
  where
    mean = cat (Dim 1) $  full' [batchSize,1,224,224] <$> [0.485 :: Float, 0.456, 0.406]
    std = cat (Dim 1) $  full' [batchSize,1,224,224] <$> [0.229 :: Float, 0.224, 0.225]

resizeImage batchSize img = expand upsampled  True [batchSize, 3,224,224] 
  where upsampled = upsample [224,224] 0 0 $ reshape [batchSize,1,28,28] img

finalLayer :: LinearSpec
finalLayer = LinearSpec 4096 10

train :: Vgg16 -> V.MnistData -> IO Linear
train vggParams trainData = do
    init <- sample finalLayer
    -- init <- sample spec
    let nImages = V.length trainData
        idxList = randomIndexes nImages
    trained <- foldLoop init numIters $
        \state iter -> do
            let idx = take batchSize (drop (iter * batchSize) idxList)
            input <- V.getImages' batchSize dataDim trainData idx
            let rszd = normalize batchSize $ resizeImage batchSize input
                label = V.getLabels' batchSize trainData idx
                loss = nllLoss' label $  logSoftmax (Dim 1) $ linear state $ vggNoFinal vggParams (1,1) (1,1) rszd
            when (iter `mod` 50 == 0) $ do
                putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
            (newParam, _) <- runStep state optimizer loss 1e-3
            pure $ replaceParameters state newParam
    pure trained
    where
      -- spec = MLPSpec (224 * 224) 64 32 10
      dataDim = 784
      numIters = 3000
      batchSize = 8 
      optimizer = GD


  -- when I switch back to MLP with resized images it still converges
mnistMain :: IO ()
mnistMain = do
    (trainData, testData) <- V.initMnist "data"

    img <- V.getImages' 1 784 trainData [1]
    model <- sample vggSpec 
    vggParams <- S.loadParams model "build/vgg16.pt" 
    model <- train vggParams trainData

    mapM (\idx -> do
        testImg <- V.getImages' 1 784 testData [idx]
        V.dispImage testImg
        let rszd = resizeImage 1 testImg
        putStrLn $ "Model        : " ++ (show . (argmax (Dim 1) RemoveDim) .
                                         Torch.exp .
                                         logSoftmax (Dim 1) .
                                         linear model $ vggNoFinal vggParams (1,1) (1,1) rszd)
        -- putStrLn $ "Model        : " ++ (show . (argmax (Dim 1) RemoveDim) .
        --                                  Torch.exp $
        --                                  mlp model rszd)

        putStrLn $ "Ground Truth : " ++ (show $ V.getLabels' 1 testData [idx])
        ) [0..10]

    putStrLn "Done"
