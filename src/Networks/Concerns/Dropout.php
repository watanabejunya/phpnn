<?php

namespace PhpNN\Foundation\Networks\Concerns;

trait Dropout
{
    /**
     * Dropout factors to multiply to the outputs of a layer.
     *
     * @var array[array[float]]
     */
    protected $dropouts = [];

    /**
     * Probability of dropping the output of a neuron. Set false if you want to
     * employ dropout into a layer.
     *
     * @var array[array[float]]
     */
    protected $dropProbabilities = [];

    /**
     * Set dropout factors.
     *
     * @param  int   $l
     * @return void
     */
    protected function setDropouts(int $l): void
    {
        if (! $this->dropProbabilities[$l] || $l === count($this->structure) - 1) {
            $this->dropouts[$l] = array_fill(0, $this->structure[$l], 1.0);
        } else {
            for ($k = 0; $k < $this->structure[$k]; $k++) {
                $this->dropout[$l][$k] = binary_rand($this->dropProbabilities[$l]) ?
                    1.0 / $this->dropProbabilities[$l] : 0.0;
            }
        }
    }
}
