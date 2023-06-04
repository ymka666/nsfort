"""Programming examples that show how to use the SinCosModel."""
# [START program]
# [START import]
import nsfort
# [END import]


def main():
    # [START ideal_quadratic_model]
    # Create ideal quadratic model for the basis
    idel_data = nsfort.ideal_quadratic_model(
        size=10000,
        coef=0.0000005
    )
    print('Ideal data:\n', idel_data)
    # [END ideal_quadratic_model]

    # [START noise_anomaly_model]
    # Create generated data with normal noise and anomalies
    # based on the created ideal
    created_data = nsfort.noise_anomaly_model(
        model=idel_data,
        mean=0,
        spread=5,
        n_anomaly=1000,
        anomaly_error=3
    )
    print('Created data:\n', created_data)
    # [END noise_anomaly_model]

    # [START TranscendentModel]
    # Create class object based on generated data
    model = nsfort.TranscendentModel(data=created_data)
    print(f'Model created. Size: {model.size}')
    print(f'Data vector: {model.data_vector}')
    # [END TranscendentModel]

    # [START approximate]
    # Approximate the data using the nonlinear sin cos method
    data_output, coefs_vector = model.approximate()
    print(f'Approximation function: {model.view()}')
    print(f'Approximation data: {data_output}')
    print(f'Approximation coefs: {coefs_vector}')
    # [END approximate]

    # [START approximate]
    # Extrapolate the data using the nonlinear sin cos method
    # extrapolation_coef is an extrapolation factor
    extrapolation_coef = 0.5
    data_output, coefs_vector = model.approximate(coef=extrapolation_coef)
    print(f'Extrapolation coef: {model.extrapolation_coef}')
    print(f'Extrapolation num: {model.extrapolation_num}')
    print(f'Extrapolation data size: {model.output_size}')
    print(f'Extrapolation data: {data_output}')
    print(f'Extrapolation coefs: {coefs_vector}')
    # [END approximate]


if __name__ == '__main__':
    main()
# [END program]
