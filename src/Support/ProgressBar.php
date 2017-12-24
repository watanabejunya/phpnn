<?php

namespace PhpNN\Foundation\Support;

class ProgressBar
{
    /**
     * Width definition.
     *
     * @var int
     */
    const BAR_WIDTH = 30;
    const DISPLAY_WIDTH = 00;

    /**
     * Format to be displayed.
     *
     * @var string
     */
    private $format = 'Time: #time  Epoch: #count/#max [#bar]  Loss: #loss  Validity: #validity%';

    /**
     * Placeholders to replace.
     *
     * @var array
     */
    private $replacements = [
        'time',
        'count',
        'max',
        'bar',
        'loss',
        'validity',
    ];

    /**
     * Time stamp of start time.
     *
     * @var int
     */
    private $startTime;

    /**
     * Current count.
     *
     * @var int
     */
    private $count;

    /**
     * Maximum count.
     *
     * @var int
     */
    private $max;

    /**
     * Loss of a network.
     *
     * @var float
     */
    private $loss;

    /**
     * Validity of a network.
     *
     * @var float
     */
    private $validity;

    /**
     * Class constructor
     */
    public function __construct(int $max = 1, int $count = 1)
    {
        $this->startTime = time();
        $this->setMax($max);
        $this->setCount($count);
    }

    /**
     * Advance progress bar by a given amount.
     *
     * @param int $amount
     * @return void
     */
    public function update(array $values): void
    {
        foreach ($values as $key => $value) {
            $this->{'set' . ucfirst($key)}($value);
        }

        $this->display();
    }

    /**
     * Advance count by given amount.
     *
     * @param int  $amount
     */
    public function advance(int $amount = 1): void
    {
        $this->update(['count' => $this->count + $amount]);
    }

    /**
     * Print the progress bar.
     *
     * @return void.
     */
    private function display(): void
    {
        $buffer = $this->format;

        foreach ($this->replacements as $replacement) {
            $buffer = str_replace('#' . $replacement, $this->{'get' . ucfirst($replacement)}(), $buffer);
        }

        if ($this->count < $this->max) {
            echo str_pad($buffer, self::BAR_WIDTH) . "\r";
        } else {
            echo str_pad($buffer, self::BAR_WIDTH) . "\n";
        }
    }

    /**
     * Getter for time.
     *
     * @return string
     */
    public function getTime(): string
    {
        $second = time() - $this->startTime;

        $hour = (int)floor($second / 3600.0);
        $minute = (int)floor(($second / 60.0) % 60);
        $second = $second % 60;

        return $hour . ':' . $minute . ':' . $second;
    }

    /**
     * Getter for count.
     *
     * @return int
     */
    public function getCount(): int
    {
        return $this->count;
    }

    /**
     * Setter for count.
     *
     * @param  int  $count
     * @return void
     */
    public function setCount(int $count): void
    {
        assert(0 < $count && $count <= $this->max);

        $this->count = $count;
    }

    /**
     * Getter for max.
     *
     * @return int
     */
    public function getMax(): int
    {
        return $this->max;
    }

    /**
     * Setter for max.
     *
     * @param  int  $max
     * @return void
     */
    public function setMax(int $max): void
    {
        assert(0 < $max);

        $this->max = $max;
    }

    /**
     * Getter for bar.
     *
     * @return int
     */
    public function getBar(): string
    {
        $position = (int)round(self::BAR_WIDTH * $this->count / (float)$this->max);
        $bar = str_pad('', $position, '-') . '>';

        return str_pad($bar, self::BAR_WIDTH, ' ');
    }

    /**
     * Getter for progress.
     *
     * @return string
     */
    public function getProgress(): string
    {
        return number_format(100.0 * $this->count / (float)$this->max);
    }

    /**
     * Getter for loss.
     *
     * @return float
     */
    public function getLoss(): float
    {
        return number_format($this->loss, 3);
    }

    /**
     * Setter for max.
     *
     * @param  float  $loss
     * @return void
     */
    public function setLoss(float $loss): void
    {
        $this->loss = $loss;
    }

    /**
     * Getter for validity.
     *
     * @return string
     */
    public function getValidity(): string
    {
        return number_format($this->validity * 100.0, 1);
    }

    /**
     * Setter for validity.
     *
     * @param  float  $validity
     * @return void
     */
    public function setValidity(float $validity): void
    {
        assert(0.0 <= $validity && $validity <= 1.0);

        $this->validity = $validity;
    }
}
