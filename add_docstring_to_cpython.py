import re

def add_docstrings_to_mojo_file(mojo_filename, reference_filename, output_filename):
    # Load reference mapping from the reference file
    reference_mapping = {}
    with open(reference_filename, 'r') as ref_file:
        for line in ref_file:
            match = re.match(r'(https://docs\.python\.org/3/c-api/.*#c\.\w+)\s+(\w+)', line)
            if match:
                reference_mapping[match.group(2)] = match.group(1)

    # Open and process the Mojo file
    with open(mojo_filename, 'r') as mojo_file:
        lines = mojo_file.readlines()

    updated_lines = []
    for i, line in enumerate(lines):
        updated_lines.append(line)
        fn_match = re.match(r'    fn (\w+)(.*)', line)
        if fn_match:
            function_name = fn_match.group(1)
            if function_name in reference_mapping:
                # Add the docstring below the function definition
                docstring = f'        """See {reference_mapping[function_name]}"""\n'
                updated_lines.append(docstring)
            else:
                pass

    # Write the updated lines to the output file
    with open(output_filename, 'w') as output_file:
        output_file.writelines(updated_lines)

# Example usage
mojo_filename = 'stdlib/src/python/_cpython.mojo'
reference_filename = 'c-abi.txt'
output_filename = 'stdlib/src/python/_cpython.mojo'
add_docstrings_to_mojo_file(mojo_filename, reference_filename, output_filename)
