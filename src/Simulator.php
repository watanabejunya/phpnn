<?php

namespace PhpNN\Foundation;

use PhpNN\Foundation\Networks\Network;
use PhpNN\Foundation\Support\ProgressBar;

abstract class Simulator
{
    /**
     * Maximum number of epoch to learn.
     *
     * @var int
     */
    protected $epoch;

    /**
     * Configuration of the neural network.
     *
     * @var array
     */
    protected $config;

    /**
     * Neural network.
     *
     * @var \PhpNN\Foundation\Networks\Network
     */
    protected $network;

    /**
     * Progress bar.
     *
     * @var \PhpNN\Foundation\Support\ProgressBar
     */
    protected $progressBar;

    /**
     * Input data set and output set (i.e. correct answer set for each input)
     * for training and testing, respectively.
     *
     * @var array
     */
    protected $trainingInputSet = [];
    protected $trainingOutputSet = [];
    protected $testingInputSet = [];
    protected $testingOutputSet = [];

    /**
     * Make a new neural network or load a cached one if exists.
     *
     *
     */
    public function __construct(array $options = [])
    {
        $this->network = new Network($this->config);
        $this->progressBar = new ProgressBar($this->epoch);

        if (! isset($options['no-cache']) && file_exists($this->modelFile)) {
            $this->load($this->modelFile);
        } else {
            $this->setup();

            $this->network->init();
        }
    }

    /**
     * Configure the neural network by adding layers.
     *
     * @return void
     */
    abstract protected function setup(): void;

    /**
     * Run simulation and return exit code.
     *
     * @return int
     */
    public function run(): int
    {
        // Set training data set.
        $this->trainingInputSet = $this->getTrainingData();
        foreach ($this->trainingInputSet as $input) {
            $this->trainingOutputSet[] = $this->getAnswer($input);
        }

        // Set testing data set.
        $this->testingInputSet = $this->getTestingData();
        foreach ($this->testingInputSet as $input) {
            $this->testingOutputSet[] = $this->getAnswer($input);
        }

        for ($i = 1; $i <= $this->epoch; $i++) {
            // Train the network to fit for the model.
            $trainingLoss = $this->network->train($this->trainingInputSet, $this->trainingOutputSet);

            // Test the network and validate the output of the network.
            [$testingLoss, $validity] = $this->network->test(
                $this->testingInputSet,
                $this->testingOutputSet,
                $this->getValidator()
            );

            $this->print($i, $trainingLoss, $testingLoss, $validity);

            $this->progressBar->update([
                'count' => $i,
                'loss' => $testingLoss,
                'validity' => $validity
            ]);
        }

        // Save the trained network as a cache.
        if (! empty($this->modelFile)) {
            $this->save($this->modelFile);
        }

        return 0;
    }

    /**
     * Get data set for training.
     *
     * @var array
     */
    abstract protected function getTrainingData(): array;

    /**
     * Get data set for testing.
     *
     * @var array
     */
    abstract protected function getTestingData(): array;

    /**
     * Calculate answer for the given input. If it is a classification problem
     * that you try to solve, this method should return array-formed value
     * like [0, ..., 1, ..., 0] or it is a continuous problems this should return
     * float value.
     *
     * @param  array  $input
     * @return array
     */
    abstract protected function getAnswer(array $input): array;

    /**
     * Get callback function to validate the result of learning.
     *
     * @return null|callable
     */
    abstract protected function getValidator(): ?callable;

    /**
     * Print the result here if you want.
     *
     * @param  int    $epoch
     * @param  float  $loss
     * @param  float  $validity
     * @return void
     */
    abstract protected function print(int $epoch, float $trainingLoss, float $testingLoss, float $validity): void;

    /**
     * Save this class as a cache.
     *
     * @param  string  $file
     * @return void
     */
    private function save(string $file): void
    {
        assert(isset($this->network));

        file_put_contents($file, serialize($this->network));
    }

    /**
     * Load cached object of this class.
     *
     * @param  string  $file
     * @return \PhpNN\Foundation\Networks\Network
     */
    private function load(string $file): Network
    {
        assert(file_exists($file));

        return unserialize(file_get_contents($file));
    }
}
