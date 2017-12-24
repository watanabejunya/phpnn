<?php

namespace PhpNN\Models;

use PhpNN\Foundation\Simulator;
use PhpNN\Foundation\Losses\MeanSquareLoss;
use PhpNN\Foundation\Neurons\{RectifierNeuron, SigmoidNeuron, LinearNeuron, TanhNeuron};

class Donut extends Simulator
{
    /**
     * File name of cached model.
     *
     * @var string
     */
    protected $modelFile = '';

    /**
     * Maximum number of epoch to learn.
     *
     * @var int
     */
    protected $epoch = 300;

    /**
     * Configuration of the neural network.
     *
     * @var array
     */
    protected $config = [
        'learningRate' => 0.005,
        'batchSize' => 16,
        'numberOfLayers' => 5,
        'inputSize' => 2,
        'outputSize' => 1,
    ];

    /**
     * Configure the neural network by adding layers.
     *
     * @return void
     */
    public function setup(): void
    {
        $this->network->setLossFunction(new MeanSquareLoss());

        $this->network->addLayer(new RectifierNeuron(), 32);
        $this->network->addLayer(new SigmoidNeuron(), 64);
        $this->network->addLayer(new RectifierNeuron(), 32);
        $this->network->addLayer(new TanhNeuron(), 1);
    }

    /**
     * Get data set for training.
     *
     * @var array
     */
    public function getTrainingData(): array
    {
        $data = [];
        for ($i = 0; $i < 5000; $i++) {
            $data[] = [gauss_rand(), gauss_rand()];
        }
        return $data;
    }

    /**
     * Get data set for testing.
     *
     * @var array
     */
    protected function getTestingData(): array
    {
        $data = [];
        for ($i = 0; $i < 500; $i++) {
            $data[] = [gauss_rand(), gauss_rand()];
        }
        return $data;
    }

    /**
     * Answer if the given input is in 1 < x^2 + y^2 < 4.
     *
     * @param  array  $input
     * @return array
     */
    public function getAnswer(array $input): array
    {
        $radius = pow($input[0], 2.0) + pow($input[1], 2.0);

        if (1.0 < $radius && $radius < 4.0) {
            return [1.0];
        }
        return [-1.0];
    }

    /**
     * Get callback function to evaluate the result of learning.
     *
     * @return null|callable
     */
    protected function getValidator(): ?callable
    {
        return function (array $output, array $answer): float {
            if ($output[0] * $answer[0] > 0.0) {
                return 1.0;
            }
            return 0.0;
        };
    }

    /**
     * Print the result here if you want.
     *
     * @param  int    $epoch
     * @param  float  $trainingLoss
     * @param  float  $testingLoss
     * @param  float  $validity
     * @return void
     */
    protected function print(int $epoch, float $trainingLoss, float $testingLoss, float $validity): void
    {
        $lossFile = __DIR__ . '/../dest/Donut/Loss.csv';
        $validityFile = __DIR__ . '/../dest/Donut/Validity.csv';
        $mapFile = __DIR__ . '/../dest/Donut/Assumption.csv';

        if ($epoch === 1) {
            file_put_contents($lossFile, '# ' . json_encode($this->config) . PHP_EOL);
            file_put_contents($validityFile, '# ' . json_encode($this->config) . PHP_EOL);
            file_put_contents($mapFile, '# ' . json_encode($this->config) . PHP_EOL);
        } elseif ($epoch === $this->epoch) {
            for ($n = 0; $n < count($this->testingInputSet); $n++) {
                $output = $this->network->calculate($this->testingInputSet[$n]);

                if ($output[0] > 0.0) {
                    file_put_contents(
                        $mapFile,
                        $this->testingInputSet[$n][0] . ', ' . $this->testingInputSet[$n][1] . PHP_EOL,
                        FILE_APPEND
                    );
                }
            }
        }

        file_put_contents($lossFile, $epoch . ', ' . $testingLoss . PHP_EOL, FILE_APPEND);
        file_put_contents($validityFile, $epoch . ', ' . $validity . PHP_EOL, FILE_APPEND);
    }
}
