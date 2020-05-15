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


finalLayer = LinearSpec 4096 10

randomIndexes :: Int -> [Int]
randomIndexes size = (`mod` size) <$> randoms seed where seed = mkStdGen 123


-- this is probably different normalization than it was trained on, and that is causing problems
normalize :: Tensor -> Tensor
normalize img = img / (asTensor (255.0 :: Float)) 

resizeImage batchSize img = expand upsampled  True [batchSize, 3,224,224] 
  where upsampled = upsample [224,224] 0 0 $ reshape [batchSize,1,28,28] img

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
            let rszd = normalize $ resizeImage batchSize input
                label = V.getLabels' batchSize trainData idx
                loss = nllLoss' label $  logSoftmax (Dim 1) $ linear state $ vgg vggParams (1,1) (1,1) rszd
            when (iter `mod` 50 == 0) $ do
                putStrLn $ "Iteration: " ++ show iter ++ " | Loss: " ++ show loss
            (newParam, _) <- runStep state optimizer loss 1e-3
            pure $ replaceParameters state newParam
    pure trained
    where
        dataDim = 784
        numIters = 200 
        batchSize = 4
        optimizer = GD


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
        let rszd = expand (upsample [224,224] 0 0 $ reshape [1,1,28,28] testImg )  True [1, 3,224,224] 
        putStrLn $ "Model        : " ++ (show . (argmax (Dim 1) RemoveDim) .  Torch.exp . logSoftmax (Dim 1) .  linear model $ vgg vggParams (1,1) (1,1) rszd)
        putStrLn $ "Ground Truth : " ++ (show $ V.getLabels' 1 testData [idx])
        ) [0..10]

    putStrLn "Done"
