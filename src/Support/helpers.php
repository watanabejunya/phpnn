<?php

if (! function_exists('debug_dump')) {
    /**
     * Dump all arguments and stop the program.
     *
     * @param  mixed
     * @return void
     */
    function debug_dump(...$values): void
    {
        foreach ($values as $value) {
            print_r($value);
        }

        die(0);
    }
}

if (! function_exists('gauss_rand')) {
    /**
     * Return a Gaussian random value.
     *
     * @return float
     */
    function gauss_rand(float $dispersion = 1.0): float
    {
        $uniform1 = mt_rand() / mt_getrandmax();
        $uniform2 = mt_rand() / mt_getrandmax();

        return $dispersion * sqrt(-2.0 * log($uniform1)) * cos(2.0 * M_PI * $uniform2);
    }
}

if (! function_exists('binary_rand')) {
    /**
     * Return 1 or 0 with the probability p.
     *
     * @return int
     */
    function binary_rand(float $p): int
    {
        assert(0 <= $p && $p <= 1.0);

        return (mt_rand() / mt_getrandmax() <= $p) ? 1 : 0;
    }
}

if (! function_exists('matrix_fill')) {
    /**
     * Create 2-dimensional array with given value.
     *
     * @param  int    $rows
     * @param  int    $column
     * @param  mixed  $value
     * @return array
     */
    function matrix_fill(int $rows, int $columns, $value): array
    {
        return array_map(function () use ($columns, $value) {
            return array_fill(0, $columns, $value);
        }, array_fill(0, $rows, []));
    }
}

if (! function_exists('array_shuffle')) {
    /**
     * Shuffle array randomly.
     *
     * @param  array  $values
     * @return array
     */
    function array_shuffle(array $values): array
    {
        shuffle($values);

        return $values;
    }
}

if (! function_exists('array_last')) {
    /**
     * Get last item of array.
     *
     * @param  array  $values
     * @return mixed
     */
    function array_last(array $values)
    {
        return end($values);
    }
}
