<?php

namespace PhpNN\Foundation\Losses;

class MeanSquareLoss implements Loss
{
    /**
     * Calculate loss (sometimes called cost) between the output of a network and the answer.
     *
     * @param  array  $output
     * @param  array  $answer
     * @return float
     */
    public function loss(array $output, array $answer): float
    {
        $loss = 0;
        for ($n = 0; $n < count($output); $n++) {
            $loss += pow($output[$n] - $answer[$n], 2.0) / 2.0 / count($output);
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
