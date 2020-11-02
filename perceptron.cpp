#include "stdio.h"
#include "math.h"
#include "algorithm"
#include "random"
#include "ctime"

#define M 5
#define N 28

// Working with arrays and matices

class Matrix
{
public:
    float **content;
    int m;
    int n;

    Matrix()
    {
        content = NULL;
        m = 0;
        n = 0;
    }

    Matrix(const int &m, const int &n) : Matrix()
    {
        content = new float *[m];
        for (int i = 0; i < m; ++i)
        {
            content[i] = new float[n];
            for (int j = 0; j < n; j++)
            {
                content[i][j] = 0;
            }
        }
        this->m = m;
        this->n = n;
    }

    // copy-swap

    Matrix(const Matrix &other) : Matrix(other.m, other.n)
    {
        for (int i = 0; i < other.m; i++)
            for (int j = 0; j < other.n; j++)
                content[i][j] = other.content[i][j];
    }

    friend void swap(Matrix &first, Matrix &second)
    {
        std::swap(first.content, second.content);
        std::swap(first.m, second.m);
        std::swap(first.n, second.n);
    }

    Matrix &operator=(Matrix matrix)
    {
        swap(*this, matrix);
        return *this;
    }

    // Overriding of math operations for matrix

    friend Matrix operator*(Matrix matrix, const float &value)
    {
        for (int i = 0; i < matrix.m; i++)
            for (int j = 0; j < matrix.n; j++)
                matrix.content[i][j] *= value;
        return matrix;
    }

    friend float* operator*(const Matrix &matrix, const int* vector)
    {
        float* result = new float[matrix.m];
        for (int i = 0; i < matrix.m; i++)
        {
            result[i] = 0;
            for (int j = 0; j < matrix.n; j++)
            {
                result[i] += matrix.content[i][j] * vector[j];
            }
        }
        return result;
    }

    friend Matrix operator-(const Matrix &matrix1, const Matrix &matrix2)
    {
        Matrix result(matrix1.m, matrix1.n);
        for (int i = 0; i < matrix1.m; i++)
            for (int j = 0; j < matrix1.n; j++)
                result.content[i][j] = matrix1.content[i][j] - matrix2.content[i][j];
        return result;
    }

    // Help functions

    // Fill matrix with random values within range [-1, 1]
    void set_random()
    {
        std::srand(std::time(NULL));
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                content[i][j] = 2.0f * (std::rand() / RAND_MAX) - 1.0f;
    }

    void print()
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                printf("%f   ", content[i][j]);
            }
            printf("\n");
        }
    }

    ~Matrix()
    {
        for (int i = 0; i < m; i++)
        {
            delete[] content[i];
        }
        delete[] content;
    }
};


// Array helper function

void print(float *arr, int count)
{
    for (int i = 0; i < count; i++)
    {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

/* Structures for neural networks */

struct TrainingInput
{
    int *input_neurons;
    float *output_neurons;
};

class NeuralNetwork
{
    Matrix matrix;
    const float al = 1.0f;

    // activation function
    float f(const float &x)
    {
        return 1.0f / (1.0f + expf(-al * x));
    }

    // Activation function derivative
    // Can be calculated by value of activation function
    float df(const float &f)
    {
        return al * f * (1 - f);
    }

public:
    const int n_inputs;
    const int n_outputs;

    NeuralNetwork(int n_inputs, int n_outputs) : n_inputs(n_inputs), n_outputs(n_outputs)
    {
        matrix = Matrix(n_outputs, n_inputs);
        matrix.set_random();
    }

    float* calculate_one(int *inputs)
    {
        float* results = matrix * inputs;
        for (int i = 0; i < n_outputs; i++)
            results[i] = f(results[i]);
        return results;
    }

    float** calculate_many(int** inputs, const int &inputs_size)
    {
        float** results = new float*[inputs_size];
        for (int i = 0; i < inputs_size; i++)
            results[i] = calculate_one(inputs[i]);
        return results;
    }

    // Neural network learning

    void learn(int** inputs, float** labels, int train_input_size, float tolerance, int max_iterations)
    {
        int iterations = 0;
        float err = tolerance + 1;
        for (int iter = 0; iter < max_iterations && err > tolerance; iter++)
        {
            float** outputs = calculate_many(inputs, train_input_size);
            // Calculate correction matrix
            Matrix correction_matrix(n_outputs, n_inputs);  // Shape is the same, as weights
            for (int i = 0; i < n_outputs; i++)
            {
                for (int j = 0; j < n_inputs; j++)
                {
                    for (int k = 0; k < train_input_size; k++)
                    {
                        correction_matrix.content[i][j] = -(labels[k][i] - outputs[k][i]) * df(outputs[k][i]) * inputs[k][j];
                    }
                }
            }
            // Correct weights
            matrix = matrix - correction_matrix;
        }
    }
};

// Main script

int main()
{
    NeuralNetwork net(28, 5);
    int image_stat[5][28] = {
        { 1,1,1,1, 1,0,0,1, 1,0,0,1, 1,0,0,1, 1,0,0,1, 1,0,0,1, 1,1,1,1 },
        { 0,0,0,1, 0,0,1,1, 0,1,0,1, 1,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1 },
        { 1,1,1,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,1,0, 0,1,0,0, 1,1,1,1 },
        { 1,1,1,1, 0,0,1,0, 0,1,0,0, 1,1,1,1, 0,0,1,0, 0,1,0,0, 1,0,0,0 },
        { 1,0,0,1, 1,0,0,1, 1,0,0,1, 1,1,1,1, 1,0,0,1, 1,0,0,1, 1,0,0,1 }
    };
    float cipher_stat[5][5] = {
        { 1,0,0,0,0 },
        { 0,1,0,0,0 },
        { 0,0,1,0,0 },
        { 0,0,0,1,0 },
        { 0,0,0,0,1 }
    };
    // TODO
    int** image = NULL;
    float** cipher = NULL;


    for (int i = 0; i < 7000; i++)
    {
        int test_idx = i % 5;
        TrainingInput tr = {image[test_idx], cipher[test_idx]};
        net.learn(image, cipher, 5, 1e-5, 7000);
    }

    for (int i = 0; i < 5; i++)
    {
        print(net.calculate_one(image[i]), net.n_outputs);
    }

    printf("Program finished.");
}