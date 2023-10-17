from json import dumps

def clean_col_name(col_name):
    col_name = col_name.strip()
    col_name = (
        col_name.replace(".1", "")
        .replace("(typhos)", "")
        .replace("yellowish", "yellow")
        .replace("yellowing", "yellow")
    )
    return col_name

def clean_col_name_food(col_name):
    col_name = col_name.strip()
    col_name = (
        col_name.replace(' ','_')
        .replace('-','_')
        .replace('/','_')
    )
    return col_name


def json_res(text:str=None, **kwargs):
    """
    Str or Dict input
    
    Str
    text: raw text message

    others:
    entries: single choise, render as buttons
    options: multiple choise, render as checklist
    table: 
      header: list of column names
      content: list of rows(list)
    suffix: raw text as tail below all other components
    **kwargs: others
    """
    local_dict = locals()
    local_dict.update(local_dict.pop('kwargs'))
    if local_dict['text'] == None:
        del local_dict['text']
    return dumps(local_dict)
