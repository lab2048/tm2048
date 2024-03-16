# How to use the repository

1. Clone the repository
2. Install the dependencies


# How to import the folder into your project(code)?

1. Locate the folder, If you put the folder under `/Users/jirlong/Dropbox/Programming`
    ```python
    import sys
    sys.path.append(r'/Users/jirlong/Dropbox/Programming')
    from folder_name import file_name
    ```
2. Locate the folder, You may put the folder under your current working directory (the project folder containing your code)
    ```python
    from folder_name import file_name
    ```
3. Reload library: if you have made changes to the library, you may need to reload the library
    ```python
    from importlib import reload
    import VISToolkit as vis
    reload(vis)

    import PTTToolkit as ptt
    reload(ptt)
    ```