<?php

namespace PhpNN\Foundation\Losses;

use PhpNN\Foundation\Neurons\Neuron;

class CrossEntropyLoss implements Loss
{
    /**
     * The neuron of output layer.
     *
     * @var \PhpNN\Foundation\Neurons\Neuron
     */
    private $neuron;

    /**
     * Set the neuron of output layer.
     *
     * @var void
     */
    public function setNeuron(Neuron $neuron): void
    {
        $this->neuron = $neuron;
    }

    /**
     * Calculate loss (sometimes called cost) between the output of a network and the answer.
     *
     * @param  array  $outputs
     * @param  array  $answers
     * @return float
     */
    public function loss(array $output, array $answer): float
    {
        $loss = 0;
        for ($n = 0; $n < count($output); $n++) {
            $loss -= $answer[$n] * log($output[$n]) + (1.0 - $answer[$n]) * log(1.0 - $output[$n]);
        }
        return $loss;
    }

    /**
     * Differential function of loss function.
     *
     * @param  array  $output
     * @param  array  $answer
     * @return array
     */
    public function differentiate(array $output, array $answer): array
    {
        $error = [];
        for ($k = 0; $k < count($output); $k++) {
            $error[] = $output[$k] - $answer[$k];
        }
        return  $error;
    }
}
