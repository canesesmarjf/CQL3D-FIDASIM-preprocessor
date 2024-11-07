function namelist = read_nml(filename)
    % Initialize output structure
    namelist = struct();

    % Open the file
    fid = fopen(filename, 'r');
    if fid == -1
        error('Could not open file.');
    end

    % Read file line-by-line
    currentGroup = '';
    while ~feof(fid)
        line = strtrim(fgetl(fid));  % Read and trim the line

        % Remove comments (anything after '!')
        line = regexprep(line, '\s*!.*', '');

        % Skip empty lines after removing comments
        if isempty(line)
            continue;
        end
        
        % Check for the start of a namelist group
        groupStart = regexp(line, '^\s*[&$](\w+)', 'tokens', 'once');
        if ~isempty(groupStart)
            currentGroup = groupStart{1};  % Get the group name
            namelist.(currentGroup) = struct();  % Initialize the group in the structure
            continue;
        end
        
        % Check for the end of a namelist group
        if strcmp(strtrim(line), '/')
            currentGroup = '';  % Reset group
            continue;
        end
        
        % Parse name-value pairs within a group
        if ~isempty(currentGroup) && contains(line, '=')
            % Enhanced regex to handle array indices and scientific notation
            tokens = regexp(line, '(\w+)(\([\d,]+\))?\s*=\s*([^\s,]+)', 'tokens');
            for i = 1:length(tokens)
                varName = tokens{i}{1};         % Variable name
                indices = tokens{i}{2};         % Optional array indices
                varValue = tokens{i}{3};        % Variable value


                % ==================================
                % Temporary fix 
                % ==================================                
                % if more than one dimension:
                if contains(indices,',')
                    continue
                end
                % ==================================
                % Temporary fix 
                % ==================================    

                % Convert varValue to number if possible, otherwise keep as a string
                numericValue = str2double(varValue);
                if ~isnan(numericValue)
                    varValue = numericValue;
                else
                    varValue = strtrim(varValue);  % Trim whitespace
                    varValue = strrep(varValue, '''', '');  % Remove quotes
                end
                
                % If array indices are provided, store as an array element
                if ~isempty(indices)
                    % Convert indices string to numeric indices
                    idx = sscanf(indices, '(%d,%d)');

                    % ==================================
                    % Temporary fix 
                    % ==================================
                    if idx == 0
                        disp('')
                        continue
                    end
                    % ==================================
                    % Temporary fix 
                    % ==================================

                    % Initialize or expand array if needed
                    if ~isfield(namelist.(currentGroup), varName) || ~iscell(namelist.(currentGroup).(varName))
                        namelist.(currentGroup).(varName) = {};
                    end
                    
                    % Assign value at the specified index
                    namelist.(currentGroup).(varName){idx(1)} = varValue;
                else
                    % Assign as a regular variable if no indices are present
                    namelist.(currentGroup).(varName) = varValue;
                end
            end
        end
    end

    % Close the file
    fclose(fid);
end