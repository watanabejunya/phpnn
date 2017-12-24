<?php

namespace PhpNN\Foundation\Neurons;

class RectifierNeuron implements Neuron
{
    /**
     * Activation function of neuron.
     *
     * @param  float  $value
     * @return float
     */
    public function activate(float $value): float
    {
        return max(0.0, $value);
    }

    /**
     * Differential function of activation function.
     *
     * @param  float  $value
     * @return float
     */
    public function differentiate(float $value): float
    {
        return ($value >= 0.0) ? 1.0: 0.0;
    }
}
