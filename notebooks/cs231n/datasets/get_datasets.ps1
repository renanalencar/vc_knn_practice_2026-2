if (-Not (Test-Path -Path "cifar-10-batches-py" -PathType Container)) {
    Write-Host "Downloading CIFAR-10 dataset..."
    Invoke-WebRequest -Uri "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" -OutFile "cifar-10-python.tar.gz"
    
    Write-Host "Extracting CIFAR-10 dataset..."
    tar -xzvf cifar-10-python.tar.gz
    
    Write-Host "Cleaning up tar file..."
    Remove-Item "cifar-10-python.tar.gz"
    
    Write-Host "Downloading ImageNet validation data..."
    Invoke-WebRequest -Uri "http://cs231n.stanford.edu/imagenet_val_25.npz" -OutFile "imagenet_val_25.npz"

    Write-Host "Done."
} else {
    Write-Host "Datasets already exist."
}
