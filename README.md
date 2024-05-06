# Prog Rock vs the world

## Description
This project aims to classify music tracks into two categories: Progressive Rock and Non Progressive Rock genres. The classification is based on audio features extracted from the tracks.

Report available [here](https://www.arpannagar.tech/Progressive-Rock-music-classification/cap6610sp24_project_final_in_absentia.pdf).



## Owners
- Arpan Nagar
- Moinak Dey
- Joseph Bensabat
- Jokent Gaza

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/arpu-nagar/Progressive-Rock-music-classification.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Navigate to the project directory:
    ```bash
    cd Progressive-Rock-music-classification
    ```
2. Run prog_rock.ipynb to download the train and test tensors first.

3. If you prefer .py files for later on (DONOT run before downloading train and test tensors)
    ```bash
    python base_model.py
    ```
4. Models area avaiable in model/ dir. 
5. For Audio Spectrogram Transformer: refer it's [README](https://github.com/BlackThompson/AST-finetuned-Shenzhen) or refer to original [repo](https://github.com/YuanGongND/ast). 


## Contributing
Contributions are welcome! If you would like to contribute to this project, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Submit a pull request to the main repository.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
