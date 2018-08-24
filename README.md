# Fully Convolutional Networks(FCNs)

These are implementations of some different Fully Convolutional Networks with Tensorflow. The FCN model was proposed in the paper
>Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

The original implementation of FCN model is at: <https://github.com/shelhamer/fcn.berkeleyvision.org>

We have implemented two FCN models. One is based on original model(See more details in above paper). Another uses a simpler architecture and is as a baseline method in paper (DVN)
>Gygli, Michael, Mohammad Norouzi, and Anelia Angelova. "Deep Value Networks Learn to Evaluate and Iteratively Refine Structured Outputs." arXiv preprint arXiv:1703.04363 (2017).


# Details

## Requirements

Python: 3.6

Tensorflow: 1.8.0

Some other packages are required, including: **numpy**(1.14.3), **scipy**(1.1.0), **Pillow**(5.2.0).

## Architectures

### FCN-32s, FCN-16s, FCN-8s

### FCN used in DVN paper

# Dataset

We used Weizmann horses dataset to test the performance of FCN model used in DVN paper, which can be downloaded in <http://www.cs.toronto.edu/~yujiali/files/data/mrseg_data_release.zip>.

# Note

The code is still being developed ...

