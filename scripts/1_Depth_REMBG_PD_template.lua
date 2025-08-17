-- PD Howler AI Bridge - Community Release
-- Complete AI depth mapping and background removal for PD Howler
-- Created by: Claude & Community
-- Version: 1.0 - Production Ready
-- 
-- ALGORITHMS SUPPORTED:
-- 1. OpenCV Depth (Fast)
-- 2. MiDaS AI Depth (Thermal colors)
-- 3. Depth Anything V2 (State-of-the-art)
-- 4. Basic Background Removal
-- 5. AI RemBG (u2net model)
-- 6. InSPyReNet (Highest quality)
--
-- REQUIREMENTS:
-- - PD Howler with Lua scripting support
-- - Python bridge installed (see installation guide)
-- - Image open in PD Howler before running

-- CONFIGURATION - This will be automatically set by setup.bat
local BRIDGE_PATH = "{{BRIDGE_PATH_PLACEHOLDER}}"

function main()
    print("=== PD Howler AI Bridge - Community Version ===")
    print("Supporting 6 AI algorithms for depth and background removal")
    
    -- Check basic requirements
    if width == nil or height == nil then
        if Dog_MessageBox then
            Dog_MessageBox("Error: No canvas available. Please create or open an image first.")
        end
        return
    end
    
    print("Canvas: " .. width .. "x" .. height)
    print("Bridge path: " .. BRIDGE_PATH)
    
    -- Simple algorithm selection using reliable ValueBox
    local algorithm = 6  -- Default to InSPyReNet (best quality)
    
    if Dog_ValueBox then
        algorithm = Dog_ValueBox("AI Processing", 
                               "Choose algorithm:\n\n" ..
                               "1 = OpenCV Depth (Fast)\n" ..
                               "2 = MiDaS AI Depth\n" ..
                               "3 = Depth Anything V2\n" ..
                               "4 = Basic RemBG\n" ..
                               "5 = AI RemBG (u2net)\n" ..
                               "6 = InSPyReNet (Best Quality)\n\n" ..
                               "Recommended: 6", 
                               1, 6, 6)
        if algorithm == nil then
            Dog_MessageBox("Cancelled", "Processing cancelled")
            return
        end
    end
    
    print("Selected algorithm: " .. algorithm)
    
    -- Save undo state
    if Dog_SaveUndo then
        Dog_SaveUndo()
        print("Undo saved")
    end
    
    -- Show processing information
    local algorithm_info = get_algorithm_info(algorithm)
    Dog_MessageBox("Starting AI Processing", 
                  "Algorithm: " .. algorithm .. "\n" ..
                  "Type: " .. algorithm_info.type .. "\n" ..
                  "Quality: " .. algorithm_info.quality .. "\n\n" ..
                  "This will:\n" ..
                  "1. Export your drawing\n" ..
                  "2. Process with AI\n" ..
                  "3. Import result to canvas\n\n" ..
                  "Click OK to continue...")
    
    -- Process with selected algorithm
    local success = run_ai_processing(algorithm)
    
    if success then
        Dog_MessageBox("Success!", 
                      "AI processing complete!\n\n" ..
                      "Algorithm: " .. algorithm_info.name .. "\n" ..
                      "Result applied to canvas.\n" ..
                      "Original saved in undo.\n\n" ..
                      algorithm_info.output_note)
    else
        Dog_MessageBox("Error", 
                      "AI processing failed.\n" ..
                      "Check console for details.\n" ..
                      "Make sure Python bridge is installed correctly.\n\n" ..
                      "Bridge path: " .. BRIDGE_PATH)
    end
end

function get_algorithm_info(algorithm)
    local info_table = {
        [1] = {
            name = "OpenCV Depth",
            type = "Enhanced depth mapping",
            quality = "Fast",
            output_note = "Colorful depth map ready for 3D!"
        },
        [2] = {
            name = "MiDaS AI Depth", 
            type = "AI depth estimation",
            quality = "High (Thermal colors)",
            output_note = "Professional thermal-style depth map!"
        },
        [3] = {
            name = "Depth Anything V2",
            type = "State-of-the-art depth",
            quality = "Highest (Green/purple)",
            output_note = "Cutting-edge AI depth mapping!"
        },
        [4] = {
            name = "Basic RemBG",
            type = "Background removal",
            quality = "Standard",
            output_note = "Background removed, ready for compositing!"
        },
        [5] = {
            name = "AI RemBG (u2net)",
            type = "AI background removal", 
            quality = "High",
            output_note = "AI-cleaned character ready for Blender!"
        },
        [6] = {
            name = "InSPyReNet",
            type = "Premium background removal",
            quality = "Highest",
            output_note = "Professional-grade background removal!"
        }
    }
    
    return info_table[algorithm] or info_table[6]
end

function run_ai_processing(algorithm)
    print("Running AI processing with algorithm " .. algorithm)
    
    -- Step 1: Export frame data
    print("Step 1: Exporting frame data...")
    local export_success = export_frame_data()
    if not export_success then
        print("Export failed!")
        return false
    end
    print("Export successful!")
    
    -- Step 2: Create AI configuration
    print("Step 2: Creating AI config...")
    local config_success = create_ai_config(algorithm)
    if not config_success then
        print("Config creation failed!")
        return false
    end
    print("Config created!")
    
    -- Step 3: Run AI processing
    print("Step 3: Running AI processing...")
    local ai_success = run_ai_bridge()
    if not ai_success then
        print("AI execution failed!")
        return false
    end
    print("AI processing completed!")
    
    -- Step 4: Import result to canvas
    print("Step 4: Importing to canvas...")
    local import_success = import_result_to_canvas()
    if import_success then
        print("Canvas import successful!")
        if Dog_Refresh then Dog_Refresh() end
        return true
    else
        print("Canvas import failed, but AI processing may have succeeded")
        print("Check Temp folder for PNG files")
        return false
    end
end

function export_frame_data()
    local export_path = BRIDGE_PATH .. "\\Temp\\input_frame.txt"
    
    -- Create directories if needed
    if Dog_ShellExe then
        Dog_ShellExe('cmd /c "if not exist "' .. BRIDGE_PATH .. '\\Temp" mkdir "' .. BRIDGE_PATH .. '\\Temp""')
    end
    
    local file = io.open(export_path, "w")
    if not file then
        print("ERROR: Cannot create export file at " .. export_path)
        print("Make sure the Python bridge path is correct!")
        print("Current bridge path: " .. BRIDGE_PATH)
        return false
    end
    
    -- Write dimensions
    file:write(width .. "\n")
    file:write(height .. "\n")
    
    -- Export pixels with progress
    local total_pixels = width * height
    local exported_pixels = 0
    
    for y = 0, height - 1 do
        for x = 0, width - 1 do
            local r, g, b = get_rgb(x, y)
            if r and g and b then
                local r255 = math.floor(r * 255)
                local g255 = math.floor(g * 255)
                local b255 = math.floor(b * 255)
                file:write(r255 .. " " .. g255 .. " " .. b255 .. "\n")
                exported_pixels = exported_pixels + 1
            else
                file:write("0 0 0\n")
            end
        end
        
        -- Progress feedback every 50 rows
        if y % 50 == 0 and y > 0 then
            local progress_pct = math.floor((y / height) * 100)
            print("Export progress: " .. progress_pct .. "%")
        end
    end
    
    file:close()
    print("Frame export complete! " .. exported_pixels .. " pixels exported.")
    return true
end

function create_ai_config(algorithm)
    local config_path = BRIDGE_PATH .. "\\Temp\\ai_config.txt"
    
    local config_file = io.open(config_path, "w")
    if not config_file then
        print("ERROR: Cannot create config file")
        return false
    end
    
    -- Map algorithm numbers to model names
    local model_names = {
        [1] = "opencv",
        [2] = "midas",
        [3] = "depth-anything-v2", 
        [4] = "basic-rembg",
        [5] = "u2net",
        [6] = "isnet-general-use"
    }
    
    local model_name = model_names[algorithm] or "isnet-general-use"
    
    config_file:write("algorithm=" .. algorithm .. "\n")
    config_file:write("model=" .. model_name .. "\n")
    config_file:write("canvas_width=" .. width .. "\n")
    config_file:write("canvas_height=" .. height .. "\n")
    config_file:close()
    
    print("AI config created: Algorithm " .. algorithm .. ", Model " .. model_name)
    return true
end

function run_ai_bridge()
    local batch_path = BRIDGE_PATH .. "\\Temp\\run_ai.bat"
    
    local batch_file = io.open(batch_path, "w")
    if not batch_file then
        print("ERROR: Cannot create batch file")
        return false
    end
    
    batch_file:write("@echo off\n")
    batch_file:write("echo AI Processing starting...\n")
    batch_file:write("cd /d \"" .. BRIDGE_PATH .. "\"\n")
    batch_file:write("python pd_howler_ai_bridge.py\n")
    batch_file:write("echo PROCESSING_COMPLETE > Temp\\status.txt\n")
    batch_file:write("echo AI processing finished!\n")
    batch_file:close()
    
    print("Launching AI processing...")
    
    if Dog_ShellExe then
        Dog_ShellExe(batch_path)
        
        -- Wait for completion with timeout
        local status_file = BRIDGE_PATH .. "\\Temp\\status.txt"
        local wait_count = 0
        local max_wait = 60  -- 2 minutes timeout
        
        print("Waiting for AI completion...")
        
        while wait_count < max_wait do
            local status = io.open(status_file, "r")
            if status then
                local content = status:read("*all")
                status:close()
                if string.find(content, "PROCESSING_COMPLETE") then
                    print("AI processing completed successfully!")
                    return true
                end
            end
            
            wait_count = wait_count + 1
            os.execute("ping -n 3 127.0.0.1 > nul")  -- 2 second wait
            
            -- Progress feedback every 10 seconds
            if wait_count % 5 == 0 then
                local elapsed = wait_count * 2
                print("AI processing... " .. elapsed .. " seconds elapsed")
            end
        end
        
        print("AI processing timeout, but may have completed")
        return true  -- Continue anyway
    else
        print("ERROR: Dog_ShellExe not available - cannot launch AI bridge")
        return false
    end
end

function import_result_to_canvas()
    local result_path = BRIDGE_PATH .. "\\Temp\\output_result.txt"
    
    local result_file = io.open(result_path, "r")
    if not result_file then
        print("No text result file found")
        
        -- Check for PNG files as alternative
        local png_files = {
            "ai_result.png",
            "depth_map.png", 
            "background_removed.png",
            "processed_image.png"
        }
        
        for _, filename in ipairs(png_files) do
            local png_path = BRIDGE_PATH .. "\\Temp\\" .. filename
            local png_file = io.open(png_path, "r")
            if png_file then
                png_file:close()
                print("Found AI result: " .. filename)
                print("AI processing successful, but automatic canvas import failed")
                return false
            end
        end
        
        print("No AI result files found")
        return false
    end
    
    -- Read result dimensions
    local result_width = tonumber(result_file:read("*line"))
    local result_height = tonumber(result_file:read("*line"))
    
    if not result_width or not result_height then
        print("Invalid result file format")
        result_file:close()
        return false
    end
    
    print("Importing " .. result_width .. "x" .. result_height .. " result to canvas...")
    
    -- Import pixel data to canvas
    local imported_pixels = 0
    for y = 0, math.min(result_height - 1, height - 1) do
        for x = 0, math.min(result_width - 1, width - 1) do
            local line = result_file:read("*line")
            if line then
                local r255, g255, b255 = line:match("(%d+) (%d+) (%d+)")
                if r255 and g255 and b255 then
                    local r = tonumber(r255) / 255
                    local g = tonumber(g255) / 255
                    local b = tonumber(b255) / 255
                    set_rgb(x, y, r, g, b)
                    imported_pixels = imported_pixels + 1
                end
            end
        end
        
        -- Progress feedback
        if y % 50 == 0 and y > 0 then
            local progress_pct = math.floor((y / result_height) * 100)
            print("Import progress: " .. progress_pct .. "%")
        end
    end
    
    result_file:close()
    print("Canvas import complete! " .. imported_pixels .. " pixels imported.")
    return true
end

-- ============================================================================
-- COMMUNITY INSTALLATION NOTES:
-- ============================================================================
-- 
-- This script is automatically configured by setup.bat
-- The {{BRIDGE_PATH_PLACEHOLDER}} will be replaced with your chosen path
-- 
-- If you need to manually configure:
-- 1. Replace {{BRIDGE_PATH_PLACEHOLDER}} with your actual bridge path
-- 2. Use double backslashes (\\) for Windows paths in Lua
-- 3. Example: "C:\\PD_Howler_AI_Bridge"
--
-- ============================================================================

-- Actually run the script!
main()
