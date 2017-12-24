<?php

namespace PhpNN\Foundation\Neurons;

interface Neuron
{
    /**
     * Activation function of neuron.
     *
     * @param  float  $value
     * @return float
     */
    public function activate(float $value): float;

    /**
     * Differential function of activation function.
     *
     * @param  float  $value
     * @return float
     */
    public function differentiate(float $value): float;
}
