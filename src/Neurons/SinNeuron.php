<?php

namespace PhpNN\Foundation\Neurons;

class SignNeuron implements Neuron
{
    /**
     * @var float
     */
    private $max;
    private $offset;

    /**
     * Construct with parameters.
     *
     * @param float  $max
     * @param float  $offset
     */
    public function __construct(float $max = 1.0, float $offset = 0.0)
    {
        $this->max = $max;
        $this->offset = $offset;
    }

    /**
     * Activation function of neuron.
     *
     * @param  float  $value
     * @return float
     */
    public function activate(float $value): float
    {
        return $this->max * (sin($value) + $this->offset);
    }

    /**
     * Differential function of activation function.
     *
     * @param  float  $value
     * @return float
     */
    public function differentiate(float $value): float
    {
        return $this->max * cos($value);
    }
}
