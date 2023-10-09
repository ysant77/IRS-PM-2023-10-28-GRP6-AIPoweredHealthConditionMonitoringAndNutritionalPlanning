def clean_col_name(col_name):
    
    col_name = col_name.strip()
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
          <th style="border: 1px solid #dddddd; padding: 8px;">Precaution 1</th>
          <th style="border: 1px solid #dddddd; padding: 8px;">Precaution 2</th>
          <th style="border: 1px solid #dddddd; padding: 8px;">Precaution 3</th>
          <th style="border: 1px solid #dddddd; padding: 8px;">Precaution 4</th>
        </tr>
      </thead>
      <tbody>
    """
    
    # Loop through the diagnosis list and create table rows
    for diagnosis in diagnosis_list:
        disease = diagnosis[0]
        probability = diagnosis[1]
        precaution1 = diagnosis[2]
        precaution2 = diagnosis[3]
        precaution3 = diagnosis[4]
        precaution4 = diagnosis[5]
        disease = disease.strip()
        probability = round(probability * 100, 2)

        html_string += f"""
        <tr>
          <td style="border: 1px solid #dddddd; padding: 8px;">{disease}</td>
          <td style="border: 1px solid #dddddd; padding: 8px;">{probability}</td>
          <td style="border: 1px solid #dddddd; padding: 8px;">{precaution1}</td>
          <td style="border: 1px solid #dddddd; padding: 8px;">{precaution2}</td>
          <td style="border: 1px solid #dddddd; padding: 8px;">{precaution3}</td>
          <td style="border: 1px solid #dddddd; padding: 8px;">{precaution4}</td>
        </tr>
        """
    
    # Close the table tags
    html_string += """
      </tbody>
    </table>
    <p>Would you like to checkout the meal plan?</p>
    <label><input type="radio" name="askMealPlan" value="Yes"> Yes</label>
    <label><input type="radio" name="askMealPlan" value="No"> No</label>
    
    """
    
    return html_string
