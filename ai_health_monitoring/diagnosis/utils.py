def clean_col_name(col_name):
    col_name = (
        col_name.replace(".1", "")
        .replace("(typhos)", "")
        .replace("yellowish", "yellow")
        .replace("yellowing", "yellow")
    )
    return col_name

def format_diagnosis_for_display(diagnosis_list):
    """
    Takes a list of diagnoses in the form [(A, 21), (B, 20)] and 
    returns an HTML string representation for UI display.
    """

    # Begin with the table header
    html_string = """
    <table style="width:100%; border-collapse: collapse;">
      <thead>
        <tr style="background-color: #f2f2f2;">
          <th style="border: 1px solid #dddddd; padding: 8px;">Disease</th>
          <th style="border: 1px solid #dddddd; padding: 8px;">Probability (%)</th>
        </tr>
      </thead>
      <tbody>
    """
    
    # Loop through the diagnosis list and create table rows
    for disease, probability in diagnosis_list:
        html_string += f"""
        <tr>
          <td style="border: 1px solid #dddddd; padding: 8px;">{disease}</td>
          <td style="border: 1px solid #dddddd; padding: 8px;">{probability}</td>
        </tr>
        """
    
    # Close the table tags
    html_string += """
      </tbody>
    </table>
    """
    
    return html_string
