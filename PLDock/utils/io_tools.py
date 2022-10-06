import os
import requests
import zipfile
import tarfile
import contextlib
import gzip
import json
import logging
import re
import ssl
import urllib
import urllib.error as URLError
import urllib3
from PLDock import log

logger =  log()
logging.getLogger("urllib3").setLevel(logging.ERROR)

def get_list(list_dir):
    """get a list from a dir of Comma-Separated files

    Args:
        list_dir (str): dir of Comma-Separated files

    Returns:
        list: all items in this files
    """
    files = [os.path.join(list_dir, i) for i in os.listdir(
        list_dir) if os.path.isfile(os.path.join(list_dir, i))]
    all_pdb = []
    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            all_pdb += [i.lower() for i in line.rstrip().split(',')]
    return all_pdb

def non_empty_file(file_path, remove_empty=True):
    if os.path.isfile(file_path):
        if os.path.getsize(file_path):
            return True
        if remove_empty:
            os.remove(file_path)

def read_list(file, skip_lines=0):
    """
    It reads a file and returns a list of lines,

    :param file: the file to read
    :param tolist: If True, the function will return a list of lines. If False, it will return a
    generator object, defaults to False (optional)
    :return: A list of lines from the file.
    """
    with open(file, 'r', encoding='utf-8') as f:
        return list(f.readlines())[skip_lines:]


def retry(func):
    """
    It takes a function as an argument, and returns a function that will call the original function up
    to three times if it fails

    :param func: The function to be decorated
    :return: The inner function is being returned.
    """
    def inner(*args, **kwargs):
        ret = func(*args, **kwargs)
        if not ret:
            max_retry = 3
            number = 0
            while number < max_retry:
                number += 1
                result = func(*args, **kwargs)
                if result:
                    break
    return inner


def dir_cwd(adir=None):
    """
    > If the argument `adir` is `None`, return the current working directory, otherwise return the
    argument `adir`

    :param adir: The directory to change to. If None, then the current working directory is returned
    :return: The current working directory if no argument is passed, otherwise the argument is returned.
    """
    return os.getcwd() if adir is None else adir


def write_list(file, alist, mode='w'):
    """
    > The function `write_list` takes a file name, a list of strings, and a mode (defaulting to 'w') and
    writes the list to the file

    :param file: the file to write to
    :param alist: the list to be written to the file
    :param mode: 'w' for write, 'a' for append, 'r' for read, defaults to w (optional)
    """
    with open(file, mode, encoding='utf-8') as f:
        f.writelines(alist)


def unzip(zip_file, out_dir):
    """
    It takes a zip file and an output directory, and extracts the contents of the zip file into the
    output directory
    
    :param zip_file: The path to the zip file you want to unzip
    :param out_dir: The directory to extract the zip file to
    :return: The names of the files that were extracted from the zip file.
    """
    with zipfile.ZipFile(zip_file) as f:
        names = f.namelist()
        for i in names:
            f.extract(i, out_dir)
    return names


def untar(in_path: str, out_dir: str = None):
    """Untar a compressed file
    Args:
        in_path (str): the path of the compressed file
        out_dir (str, optional): The dir to put the untared files.
            Defaults to None,
            meaning put untared files in the same directory with the tar file.
    """
    if not out_dir:
        out_dir = os.path.dirname(in_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with tarfile.open(in_path) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=out_dir)
        return 1


def ungz(in_path: str, out_path: str = None):
    if not out_path:
        out_path = os.path.splitext(in_path)[0]
    with gzip.open(in_path, "rb") as gz:
        with open(out_path, "wb") as out:
            out.writelines(gz)
            return 1


# -------------------------------------------------------------------------------------------------------------------------------------
# Following https://stackoverflow.com/questions/65910282/jsondecodeerror-invalid-escape-when-parsing-from-python
# -------------------------------------------------------------------------------------------------------------------------------------
# It replaces all instances of `,]` with `]` and all instances of `\` with `\`
class LazyDecoder(json.JSONDecoder):
    def decode(self, s, **kwargs):
        regex_replacements = [
            (re.compile(r'([^\\])\\([^\\])'), r'\1\\\\\2'),
            (re.compile(r',(\s*])'), r'\1'),
        ]
        for regex, replacement in regex_replacements:
            s = regex.sub(replacement, s)
        return super().decode(s, **kwargs)


def get_url_content(url):
    """
    It takes a URL as input and returns the content of the URL as output

    :param url: The URL to be scraped
    :return: The content of the url.
    """
    #context = ssl._create_unverified_context()
    #try:
    #    #with urllib.request.urlopen(url, context=context) as info:
    #    #info = urllib.request.urlopen(url, context=context)
    #    info = urllib.request.urlopen(url)
    #    out =  info.read()
    #except urllib.error.URLError:
    #    out = 0
    out = 0
    if not out:
        http = urllib3.PoolManager()
        try:
            r = http.request('GET', url)
            #r = requests.get(url)
            if r and r.status == 200:
                # if r.status_code == 200:
                out = r.data
                # return r.content
        except Exception:
            out = 0
    if not out:
        try:
            response = requests.get(url)
            out = response.content
        except Exception:
            out = 0
    return out


def get_dict(url):
    """
    It takes a url, gets the content of the url, and then tries to decode the content into a dictionary

    :param url: the url to get the content from
    :return: A dictionary of the JSON data.
    """
    content = get_url_content(url)
    if content:
        try:
            # url_string = content.decode('unicode_escape')
            # url_string = json.loads(url_string, cls=LazyDecoder, strict=False)
            url_string = json.loads(content.decode('utf-8'))
            return url_string
        except json.decoder.JSONDecodeError:
            return


def download(url, file_path):
    """
    It downloads a file from a given url and saves it to a given file path.

    :param url: the url of the file to download
    :param file_path: The path to the file you want to download
    :return: 1
    """
    if not non_empty_file(file_path):
        cont = get_url_content(url)
        if cont:
            with contextlib.suppress(IOError):
                with open(file_path, 'wb') as f:
                    # f.write(requests.get(url).content)
                    f.write(cont)
    if not non_empty_file(file_path):
        with contextlib.suppress(IOError):
            os.system(f'wget -q -O {file_path} {url}')
    if not non_empty_file(file_path):
        with contextlib.suppress(IOError):
            os.popen(f'curl -s -f -o {file_path} {url}')
    if non_empty_file(file_path):
        return 1


def download_tar(data_dir, url: str, file_name: str, untar_file: bool = False):
    """Download from a url
    Args:
        url (str): the url of the file
        file_name (str): to set the name of the target file
        untar (bool, optional): Untar the compressed file or not.
            Defaults to True.
    """
    file_path = os.path.join(data_dir, file_name)
    if not non_empty_file(file_path):
        download(url, file_path)
        logger.info(f'Downloading {file_path} from {url} : DONE')
    else:
        logger.warning(f'{file_path} already exists, skipping ...')

    if non_empty_file(file_path) and untar_file:
        untar(file_path)


def check_download(file_path, url):
    """Check whether a file exists. If not, download it.

    Args:
        file_path (str): the path of the file
        url (str): url of the file

    Returns:
        1 : the file exists or was downloaded successfully
        None : the file does not exist and the download is unsuccessful
    """
    if not non_empty_file(file_path):
        file_dir = os.path.dirname(file_path)
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
        return download(url, file_path)
    return 1


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DIPS-Plus
#    (https://github.com/amorehead/DIPS-Plus):
# -------------------------------------------------------------------------------------------------------------------------------------
def slice_list(input_list, size):
    """
    It takes a list and a size, and returns a list of lists, where each list is the size specified

    :param input_list: The list to be sliced
    :param size: the number of chunks to divide the input list into
    :return: A list of lists, where each list is a slice of the original list.
    """
    input_size = len(input_list)
    slice_size = input_size // size
    remain = input_size % size
    result = []
    iterator = iter(input_list)
    for i in range(size):
        result.append([])
        for _ in range(slice_size):
            result[i].append(iterator.__next__())
        if remain:
            result[i].append(iterator.__next__())
            remain -= 1
    return result


def concat_lists(alists, remove_repetition=False):
    """
    It takes a list of lists and returns a list of all the elements in the lists

    :param alists: a list of lists
    :param remove_repetition: if True, the function will remove any repetition in the list, defaults to
    False (optional)
    :return: A list of all the elements in the list of lists.
    """
    new_list = []
    for i in alists:
        if remove_repetition:
            for j in i:
                if j not in new_list:
                    new_list.append(j)
        else:
            new_list += i
    return new_list


def combine_files(file_list, out_file=None, skip_lines=0):
    """
    It takes a list of files and combines them into a single file

    :param file_list: a list of files to combine
    :param out_file: the name of the file to write to. If not specified, the first file in the list is
    used
    """
    nwlines = []
    for file in file_list:
        lines = read_list(file, skip_lines)
        lines[-1] = f'{lines[-1].rstrip()}\n'
        nwlines += lines
    if out_file:
        write_list(out_file, nwlines)
    return nwlines


def dir_file_list(file_dir):
    """
    It takes a directory as input and returns a list of all the files in that directory

    :param file_dir: the directory where the files are located
    :return: A list of all the files in the directory.
    """
    return [os.path.join(file_dir, i) for i in os.listdir(
        file_dir) if os.path.isfile(os.path.join(file_dir, i))]


def spit_lines(lines, sep=',', clums='all'):
    """
    It takes a list of strings, and returns a list of strings

    :param lines: the lines to be split
    :param sep: The separator to use when splitting the lines, defaults to , (optional)
    :return: A list of strings
    """
    if clums == 'all':
        if sep is None:
            return [line.rstrip().split() for line in lines]
        return [line.rstrip().split(sep) for line in lines]
    else:
        if sep is None:
            return [[line.rstrip().split()[i] for i in clums] for line in lines]
        return [[line.rstrip().split(sep)[i] for i in clums] for line in lines]


def dir_lines(file_dir,out_file=None):
    """get a list from a dir of Comma-Separated files

    Args:
        list_dir (str): dir of Comma-Separated files

    Returns:
        list: all items in this files
    """
    files = dir_file_list(file_dir)
    return combine_files(files,out_file)


def dir_split_lines(file_dir, sep=',', clums='all'):
    """
    `dir_split_lines` takes a file directory and a separator and returns a list of lists of the lines of
    the file split by the separator

    :param file_dir: the directory of the file you want to read
    :param sep: the separator between the values in the file, defaults to , (optional)
    :return: A list of lists.
    """
    lines = dir_lines(file_dir)
    return spit_lines(lines, sep, clums)


def split_file(file, sep=',', clums='all', skip_lines=0):
    """
    > Read a file, split each line into a list of strings, and return a list of lists of strings

    :param file: the name of the file to read
    :param sep: The character used to separate the values in the file, defaults to , (optional)
    :return: A list of lists.
    """
    lines = read_list(file, skip_lines)
    return spit_lines(lines, sep, clums)


def converse_dict(adict):
    """
    For each value in the dictionary, create a key in the new dictionary with that value, and then add
    the key from the original dictionary to the list of values for the new dictionary

    :param adict: a dictionary
    :return: A dictionary with the values of the original dictionary as keys and the keys of the
    original dictionary as values.
    """
    new_dict = {i: [] for i in concat_lists(adict.values(), True)}
    for m, n in adict.items():
        for j in n:
            new_dict[j].append(m)
    return new_dict


def write_dict(file, adict, seps=('\t', ' ')):
    """
    It takes a dictionary and writes it to a file, with the keys and values separated by a tab

    :param adict: a dictionary
    :param seps: a tuple of two strings, the first is the separator between the key and the values, the
    second is the separator between the values
    """
    write_list(
        file, [f'{i}{seps[0]}{seps[1].join(j)}\n' for i, j in adict.items()])


def read_dict(file, sep=',', skip_lines=0, clums='all', repeat =False):
    """
    It takes a file and returns a dictionary where the keys are the first item in each line and the
    values are the rest of the items in each line

    :param file: the file to read from
    :return: A dictionary with the first item in each line as the key and the rest of the items as the
    value.
    """
    lines = split_file(file, sep, clums, skip_lines)
    if not repeat:
        return {i[0]: i[1:] for i in lines}
    odict = {i[0]: [] for i in lines}
    for line in lines:
        odict[line[0]] += line[1:]
    return odict


def check_dir(adir):
    if not os.path.exists(adir):
        os.makedirs(adir)
