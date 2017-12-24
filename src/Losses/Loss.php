<?php

namespace PhpNN\Foundation\Losses;

interface Loss
{
    /**
     * Calculate loss (sometimes called cost) between the output of a network and the answer.
     *
     * @param  array  $output
     * @param  array  $answer
     * @return float
     */
    public function loss(array $output, array $answer): float;

    /**
     * Differential function of loss function.
     *
     * @param  array  $output
     * @param  array  $answer
     * @return array
     */
    public function differentiate(array $output, array $answer): array;
}
