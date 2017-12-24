# PhpNN

A simple neural network library written by PHP.

## Setup

```sh
$ composer install
```

## Usage

You can easily use NN function approximation with php.

```sh
$ php phpnn `your_model_name`
```

## Customize

You can also make a new NN model as you like.

```php
<?php

namespace PhpNN\Models;

class YourModel extends Simulator
{
    /**
     * Maximum number of epoch to learn.
     *
     * @var int
     */
    protected $epoch = 500;

    /**
     * Configuration of the neural network.
     *
     * @var array
     */
    protected $config = [
        'learningRate' => 0.003,
        'batchSize' => 16,
        'numberOfLayers' => 5,
        'inputSize' => 16,
        'outputSize' => 16,
    ];

    /**
     * Configure the neural network by adding layers.
     *
     * @return void
     */
    public function setup(): void
    {
        // Set loss function.
        $this->network->setLossFunction(new MeanSquareLoss());

        // Set layers and activation functions of each layer.
        $this->network->addLayer(new RectifierNeuron(), 16);
        $this->network->addLayer(new SigmoidNeuron(), 64);
        $this->network->addLayer(new RectifierNeuron(), 32);
        $this->network->addLayer(new TanhNeuron(), 16);
    }

    // You must implement a function to provide data set for training.
    protected function getTrainingData(): array
    {
        //
    }    

    // You must implement a function to provide data set for testing.
    protected function getTestingData(): array
    {
        //
    }

    // You must implement a function to provide an answer for a given data.
    protected function getAnswer(array $input): array
    {
        //
    }

    // You must implement a function to provide a callback to validate an output of NN.
    protected function getValidator(): ?callable
    {
        //
    }
}
```

Then you need to register an alias of the model class in `config/classmap.php`.

```php
return [
    'alias' => \PhpNN\Models\YourModel::class,
];
```

Now you can execute NN simulator by the following command.

```sh
$ php phpnn alias
```

That's all. Good luck.
