#include <iostream>
#include <omp.h>

using namespace std;

void sequentialBubbleSort(int *, int);
void parallelBubbleSort(int *, int);
void swap(int &, int &);

void sequentialBubbleSort(int *a, int n)
{
    int swapped;
    for (int i = 0; i < n; i++)
    {
        swapped = 0;
        for (int j = 0; j < n - 1; j++)
        {
            if (a[j] > a[j + 1])
            {
                swap(a[j], a[j + 1]);
                swapped = 1;
            }
        }
        if (!swapped)
            break;
    }
}

void parallelBubbleSort(int *a, int n)
{
    int swapped;
    for (int i = 0; i < n; i++)
    {
        swapped = 0;
        int first = i % 2;
#pragma omp parallel for shared(a, first)
        for (int j = first; j < n - 1; j += 2)
        {
            if (a[j] > a[j + 1])
            {
                swap(a[j], a[j + 1]);
                swapped = 1;
            }
        }
        if (!swapped)
            break;
    }
}

void swap(int &a, int &b)
{
    int test = a;
    a = b;
    b = test;
}

int main()
{
    int *a, *b, n;
    cout << "\n enter total no of elements=>";
    cin >> n;
    a = new int[n];
    b = new int[n]; // Allocate a second array for parallel sort

    cout << "\n enter elements=>";
    for (int i = 0; i < n; i++)
    {
        cin >> a[i];
        b[i] = a[i]; // Copy original elements for parallel sort
    }

    double start_time = omp_get_wtime();
    sequentialBubbleSort(a, n);
    double end_time = omp_get_wtime();

    cout << "\n sorted array by sequential=>";
    for (int i = 0; i < n; i++)
    {
        cout << a[i] << endl;
    }

    cout << "Time taken by sequential algorithm: " << end_time - start_time << " seconds" << endl;

    start_time = omp_get_wtime();
    parallelBubbleSort(b, n);
    end_time = omp_get_wtime();

    cout << "\n sorted array by parallel=>";
    for (int i = 0; i < n; i++)
    {
        cout << b[i] << endl;
    }

    cout << "Time taken by parallel algorithm: " << end_time - start_time << " seconds" << endl;

    delete[] a;
    delete[] b;

    return 0;
}
