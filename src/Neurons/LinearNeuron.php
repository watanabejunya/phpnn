<?php

namespace PhpNN\Foundation\Neurons;

class LinearNeuron implements Neuron
{
    /**
     * Activation function of neuron.
     *
     * @param  float  $value
     * @return float
     */
    public function activate(float $value): float
    {
        return $value;
    }

    /**
     * Differential function of activation function.
     *
     * @param  float  $value
     * @return float
     */
    public function differentiate(float $value): float
    {
        return 1.0;
    }
}
