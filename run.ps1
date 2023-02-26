docker run -it `
    -v ${PWD}:/home/ubuntu `
    --gpus all `
    --name=faceswap `
    faceswap `
    bash