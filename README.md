# Deep-Learning-using-PyTorch
Repo contains overview of DL and why and How PyTorch is used with usecases and executed project

{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNAY+rPnGNkMxr0rwsV8Z9L",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/priyankak17/Deep-Learning-using-PyTorch/blob/main/PyTorch_DL_Overview_00.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Deep Learning with PyTorch "
      ],
      "metadata": {
        "id": "npjI_wmZaFGF"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Limitations of Deep Learning\n",
        "\n",
        "\n",
        "*   Model Explainability [Refer to papers [1.](https://scholar.google.co.in/scholar?q=deep+learning+explainability&hl=en&as_sdt=0&as_vis=1&oi=scholart)]\n",
        "\n",
        "*   Unpredictable outputs leading to unknown errors\n",
        "*   DL model require much larger dataset\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "nKJfnQyNMjiF"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Machine Learning \n",
        "\n",
        "\n",
        "*   Deals with structured data\n",
        "*   XGBoost algorithm typically used in Datascience competitions and in productions \n",
        "*   Other algorithms used are random forest, tree based algorithm, etc.\n",
        "\n",
        "Algorithms:\n",
        "\n",
        "\n",
        "\n",
        "1.   Random Forest\n",
        "2.   Gradient Boosted Models\n",
        "\n",
        "1.   Naive Bayes\n",
        "2.   Nearest Neighbour\n",
        "\n",
        "1.   Support Vector Machines\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "##Deep Learning \n",
        "\n",
        "\n",
        "\n",
        "*   Better used for unstructured data \n",
        "*   Using Tensor unstructured data like images can be turned into a structure\n",
        "\n",
        "Algorithms:\n",
        "\n",
        "\n",
        "\n",
        "1.   Neural Network\n",
        "\n",
        "1.   Fully connected Neural Network\n",
        "2.   Convolution Neural Network\n",
        "\n",
        "2.   Recurrent Neural Network\n",
        "\n",
        "1.   Transformer\n",
        "\n",
        "PyTorch deals with 1st 3 algorithms\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "SMSPAMAAN-zR"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Neural Network\n",
        "INPUT LAYER --> HIDDEN LAYER --> OUTPUT LAYER\n",
        "\n",
        "Input can be images, audio, text, which will encoded into computer readable format - Numeric form, then features/weights will be adjusted to recognise that image/text\n",
        "\n",
        "eg: Resnet 152"
      ],
      "metadata": {
        "id": "OsHSU2R4ebfw"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Types of Learning\n",
        "\n",
        "\n",
        "1.   Supervised\n",
        "1.   UnSupervised\n",
        "2.   Transfer\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "oxInTQX-ic9S"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##[PyTorch](https://pytorch.org/) \n",
        "\n",
        "Popular DL framework used by AI Agriculture, tesla autpilot, META AI, Microsoft, openAI etc.\n"
      ],
      "metadata": {
        "id": "UG4env-TlLoZ"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##[Tensor](https://deepai.org/machine-learning-glossary-and-terms/tensor)\n",
        "\n",
        "The numerically encoded data that is converted from raw data(image/audio/text) \n",
        "is called tensor. \n",
        "PyTorch is based on torch dot tensor.\n",
        "\n",
        "Ref: https://github.com/mrdbourke/pytorch-deep-learning\n"
      ],
      "metadata": {
        "id": "53HG4eRBoADK"
      }
    }
  ]
}
