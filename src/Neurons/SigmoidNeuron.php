<?php

namespace PhpNN\Foundation\Neurons;

class SigmoidNeuron implements Neuron
{
    /**
     * Activation function of neuron.
     *
     * @param  float  $value
     * @return float
     */
    public function activate(float $value): float
    {
        return 1.0 / (1.0 + exp(- $value));
    }

    /**
     * Differential function of activation function.
     *
     * @param  float  $value
     * @return float
     */
    public function differentiate(float $value): float
    {
        return $this->activate($value) * (1.0 - $this->activate($value));
    }
}
