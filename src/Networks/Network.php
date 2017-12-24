<?php

namespace PhpNN\Foundation\Networks;

use PhpNN\Foundation\Networks\Concerns\Dropout;
use PhpNN\Foundation\Neurons\Neuron;
use PhpNN\Foundation\Losses\Loss;

class Network
{
    use Dropout;

    /**
     * The number of layers of the network.
     *
     * @var int
     */
    protected $numberOfLayers;

    /**
     * Learning rate.
     *
     * @var float
     */
    protected $learningRate;

    /**
     * The size of mini batch.
     *
     * @var int
     */
    protected $batchSize;

    /**
     * The size of input of network.
     *
     * @var int
     */
    protected $inputSize;

    /**
     * The size of output of network.
     *
     * @var int
     */
    protected $outputSize;

    /**
     * Definition of network structure that shows which layer has how many neurons.
     *
     * @var array[float]
     */
    protected $structure = [];

    /**
     * Neuron types of each layer. Note that null is given for the input layer.
     *
     * @var array[\PhpNN\Foundation\Neurons\Neuron]
     */
    protected $neurons = [];

    /**
     * Loss (cost) function to calculate accuracy of the network.
     *
     * @var \PhpNN\Foundation\Losses\Loss
     */
    protected $lossFunction;

    /**
     * Weights of each input of each neuron, which are usually denoted by w_jk^(l).
     *
     * @var array[array[array[float]]]
     */
    protected $weights = [];

    /**
     * Biases of each neuron, which are usually denoted by b_j^(l).
     *
     * @var array[array[float]]
     */
    protected $biases = [];

    /**
     * Inputs of each neuron, which are usually denoted by u_j^(l).
     *
     * @var array[array[float]]
     */
    protected $inputs = [];

    /**
     * Outputs of each neuron, which are usually denoted by z_j^(l).
     * Values of the last layer means output of the network.
     *
     * @var array[array[float]]
     */
    protected $outputs = [];

    /**
     * Errors for the output of each neuron, which are usually denoted by δ_j^(l).
     *
     * @var array[array[float]]
     */
    protected $errors = [];

    /**
     * Gradient of loss function for each weight, which means ∂C / ∂w_jk^(l).
     *
     * @var array[array[array[float]]]
     */
    protected $weightGradients = [];

    /**
     * Gradient of loss function for each bias, which means ∂C / ∂b_j^(l).
     *
     * @var array[array[float]]
     */
    protected $biasGradients = [];

    /**
     * Construct with parameters. Description of each parameter is given bellow.
     *
     * @param array  $config
     */
    public function __construct(array $config = [])
    {
        assert(! isset($config["learningRate"]) || is_float($config["learningRate"]));
        assert(isset($config["inputSize"]) && is_int($config["inputSize"]));
        assert(isset($config["outputSize"]) && is_int($config["outputSize"]));
        assert(isset($config["numberOfLayers"]) && is_int($config["numberOfLayers"]));

        $this->learningRate = $config['learningRate'] ?? 0.01;
        $this->batchSize = $config['batchSize'] ?? 1;
        $this->numberOfLayers = $config['numberOfLayers'] ?? 0;
        $this->inputSize = $config['inputSize'] ?? 0;
        $this->outputSize = $config['outputSize'] ?? 0;

        // Add input layer.
        $this->structure[] = $this->inputSize;
        $this->neurons[] = null;
        $this->weights[] = matrix_fill($this->inputSize, array_last($this->structure), 0.0);
        $this->biases[] = array_fill(0, $this->inputSize, 0.0);
        $this->inputs[] = array_fill(0, $this->inputSize, 0.0);
        $this->outputs[] = array_fill(0, $this->inputSize, 0.0);
        $this->errors[] = array_fill(0, $this->inputSize, 0.0);
        $this->biasGradients[] = array_fill(0, $this->inputSize, 0.0);
        $this->weightGradients[] = matrix_fill($this->inputSize, array_last($this->structure), 0.0);
        $this->dropProbabilities[] = 0.0;
        $this->dropouts[] = array_fill(0, $this->inputSize, 1.0);
    }

    /**
     * Set loss function on the network.
     *
     * @param  \PhpNN\Foundation\Losses\Loss  $lossFunction
     * @return void
     */
    public function setLossFunction(Loss $lossFunction): void
    {
        $this->lossFunction = $lossFunction;
    }

    /**
     * Add a hidden or output layer. If you want to apply dropout method,
     * please designate dropout probability.
     *
     * @param  \PhpNN\Foundation\Neurons\Neuron  $neuron
     * @param  int                               $number
     * @return void
     */
    public function addLayer(Neuron $neuron, int $number, array $options = []): void
    {
        assert(empty($options['dropout']) || is_numeric($options['dropout']));

        // Init weights and biases with random numbers.
        $this->weights[] = matrix_fill($number, array_last($this->structure), 0.0);
        $this->biases[] = array_fill(0, $number, 0.0);
        $this->inputs[] = array_fill(0, $number, 0.0);
        $this->outputs[] = array_fill(0, $number, 0.0);
        $this->errors[] = array_fill(0, $number, 0.0);
        $this->biasGradients[] = array_fill(0, $number, 0.0);
        $this->weightGradients[] = matrix_fill($number, array_last($this->structure), 0.0);
        $this->structure[] = $number;
        $this->neurons[] = $neuron;
        $this->dropProbabilities[] = $options['dropout'] ?? 0.0;
        $this->dropouts[] = array_fill(0, $number, 1.0);
    }

    /**
     * Init weights and biases with random values.
     *
     * @return void
     */
    public function init(): void
    {
        for ($l = 1; $l < $this->numberOfLayers; $l++) {
            for ($j = 0; $j < $this->structure[$l]; $j++) {
                $this->biases[$l][$j] = gauss_rand();

                for ($k = 0; $k < $this->structure[$l-1]; $k++) {
                    $this->weights[$l][$j][$k] = gauss_rand();
                }
            }
        }
    }

    /**
     * Let the neural network start learning. Flow is described bellow.
     * 1. Input random training data and propagate forward.
     * 2. Calculate errors of each neuron by back propagation.
     * 3. Update weights and biases if count amount to batch size.
     *
     * @param  array  $inputSet
     * @param  array  $answerSet
     * @return float
     */
    public function train(array $inputSet, array $answerSet): float
    {
        assert($this->numberOfLayers === count($this->structure));
        assert(count($inputSet) > 0 && count($inputSet) === count($answerSet));
        assert($this->inputSize === count($inputSet[0]));
        assert($this->outputSize === count($answerSet[0]));

        $L = $this->numberOfLayers - 1;
        $count = 0;
        $loss = 0.0;

        foreach (array_shuffle(array_keys($inputSet)) as $n) {
            $count++;

            $this->forwardPropagate($inputSet[$n]);

            $this->backwardPropagate($answerSet[$n]);

            $loss += $this->lossFunction->loss($this->outputs[$L], $answerSet[$n])
                / (float)$this->batchSize;

            if ($count >= $this->batchSize) {
                $this->update();
                break;
            }
        }

        return $loss;
    }

    /**
     * Let the neural network start testing in the same way as learning.
     * 1. Input random training data and propagate forward.
     * 2. Calculate loss function for the output.
     * 3. Evaluate by a given callback function.
     *
     * @param  array          $inputSet
     * @param  array          $answerSet
     * @param  null|callable  $validator
     * @return array
     */
    public function test(array $inputSet, array $answerSet, callable $validator = null): array
    {
        assert(count($inputSet) > 0 && count($inputSet) === count($answerSet));
        assert($this->inputSize === count($inputSet[0]));
        assert($this->outputSize === count($answerSet[0]));

        $L = $this->numberOfLayers - 1;
        $loss = 0.0;
        $validity = 0.0;

        for ($n = 0; $n < count($answerSet); $n++) {
            $this->forwardPropagate($inputSet[$n]);

            $loss += $this->lossFunction->loss($this->outputs[$L], $answerSet[$n])
                / (float)count($answerSet);

            if (is_callable($validator)) {
                $validity += $validator($this->outputs[$L], $answerSet[$n])
                    / (float)count($answerSet);
            }
        }

        return [$loss, $validity];
    }

    /**
     * Simply calculate the output by forward propagation.
     *
     * @param  array  $input
     * @return array
     */
    public function calculate(array $input): array
    {
        assert($this->inputSize === count($input));

        $this->forwardPropagate($input);

        return $this->outputs[$this->numberOfLayers - 1];
    }

    /**
     * Propagate forward. First, we calculate the input of a neuron
     * u_j^(l) = Σ w_jk^(l) * z_k^(l-1) + b_j^(l). Then we obtain the output
     * by applying activation function of each neuron as z_j^(l) = f(u_j^(l)).
     * This propagation process continues from l = 1 to l = L.
     *
     * @param  array  $input
     * @return void
     */
    protected function forwardPropagate(array $input): void
    {
        $this->outputs[0] = $input;

        for ($l = 1; $l < $this->numberOfLayers; $l++) {
            $this->setDropouts($l);

            for ($j = 0; $j < $this->structure[$l]; $j++) {
                $this->inputs[$l][$j] = $this->biases[$l][$j];
                for ($k = 0; $k < $this->structure[$l-1]; $k++) {
                    $this->inputs[$l][$j] += $this->weights[$l][$j][$k]
                        * $this->outputs[$l-1][$k];
                }

                $this->outputs[$l][$j] = $this->dropouts[$l][$j]
                    * $this->neurons[$l]->activate($this->inputs[$l][$j]);
            }
        }
    }

    /**
     * Propagate backward. First, we calculate the errors in the output layer
     * δ_j^(L) = Σ w_jk^(l) * z_k^(l-1) + b_j^(l). Then we obtain the output
     * by applying activation function of each neuron as z_j^(l) = f(u_j^(l)).
     * This propagation process continues from l = L - 1 to l = 1.
     *
     * @param  array  $output
     * @return void
     */
    protected function backwardPropagate(array $output): void
    {
        $L = $this->numberOfLayers - 1;

        // Calculate the errors in the output layer.
        $this->errors[$L] = $this->lossFunction->differentiate($this->outputs[$L], $output);

        for ($j = 0; $j < $this->structure[$L]; $j++) {
            $this->biasGradients[$L][$j] += $this->errors[$L][$j] / (float)$this->batchSize;

            for ($k = 0; $k < $this->structure[$L-1]; $k++) {
                $this->weightGradients[$L][$j][$k] += $this->outputs[$L-1][$k]
                    * $this->errors[$L][$j] / (float)$this->batchSize;
            }
        }

        // Start back propagation.
        for ($l = $L-1; $l > 0; $l--) {
            for ($j = 0; $j < $this->structure[$l]; $j++) {
                $this->errors[$l][$j] = 0.0;
                for ($m = 0; $m < $this->structure[$l+1]; $m++) {
                    $this->errors[$l][$j] += $this->errors[$l+1][$m]
                        * $this->weights[$l+1][$m][$j]
                        * $this->neurons[$l]->differentiate($this->inputs[$l][$j]);
                }

                $this->biasGradients[$l][$j] += $this->errors[$l][$j] / (float)$this->batchSize;

                for ($k = 0; $k < $this->structure[$l-1]; $k++) {
                    $this->weightGradients[$l][$j][$k] += $this->outputs[$l-1][$k] * $this->errors[$l][$j]
                        / (float)$this->batchSize;
                }
            }
        }
    }

    /**
     * Update weight and biases in the gradient direction as bellow.
     * w_jk^(l) -> w_jk^(l) - ∂C / ∂w_jk^(l) = w_jk^(l) - σ * z_k^(l-1) * δ_j^(l)
     * b_j^(l) -> b_j^(l) - ∂C / ∂b_j^(l) = b_j^(l) - σ * δ_j^(l)
     *
     * @return void
     */
    protected function update(): void
    {
        for ($l = 1; $l < $this->numberOfLayers; $l++) {
            for ($j = 0; $j < $this->structure[$l]; $j++) {
                $this->biases[$l][$j] -= $this->learningRate * $this->biasGradients[$l][$j];
                $this->biasGradients[$l][$j] = 0.0;

                for ($k = 0; $k < $this->structure[$l-1]; $k++) {
                    $this->weights[$l][$j][$k] -= $this->learningRate * $this->weightGradients[$l][$j][$k];
                    $this->weightGradients[$l][$j][$k] = 0.0;
                }
            }
        }
    }

    /**
     * Get output of the designated layer. When no layer is specified, this
     * returns output of the output layer.
     *
     * @param  int  $l
     * @return array
     */
    public function getOutput(int $l = null): array
    {
        assert(is_null($l) || (0 <= $l && $l < $this->numberOfLayers));

        $l = $l ?? $this->numberOfLayers - 1;

        return $this->outputs[$l];
    }

    /**
     * Get error of the designated layer. When no layer is specified, this
     * returns error of the output layer.
     *
     * @param  int  $l
     * @return array
     */
    public function getError(int $l = null): array
    {
        assert(is_null($l) || (0 <= $l && $l < $this->numberOfLayers));

        $l = $l ?? $this->numberOfLayers - 1;

        return $this->errors[$l];
    }
}
