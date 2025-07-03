from http.server import HTTPServer, BaseHTTPRequestHandler
from email.parser import BytesParser
from email.policy import default
import os
import io
import csv
import random
import re
import traceback
import math
from urllib.parse import parse_qs, urlparse
from collections import defaultdict
from statistics import mean, stdev

UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

def sanitize_filename(filename):
    """Remove potentially dangerous characters from filename"""
    return re.sub(r'[^\w\-_. ]', '', filename)

def calculate_mean(values):
    """Calculate mean without numpy"""
    return mean(values) if values else 0.0

def calculate_std(values, mean_val):
    """Calculate standard deviation without numpy"""
    if len(values) < 2:
        return 0.0
    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

def analyze_column(values):
    """Analyze a column to determine its type and characteristics without numpy"""
    numeric_count = 0
    unique_values = set()
    numeric_vals = []
    
    for val in values:
        if val.strip():
            unique_values.add(val)
            try:
                num_val = float(val)
                numeric_count += 1
                numeric_vals.append(num_val)
            except ValueError:
                pass
    
    # Determine column type
    if numeric_count / len(values) > 0.8:  # Mostly numeric
        if numeric_vals:
            mean_val = calculate_mean(numeric_vals)
            std_val = calculate_std(numeric_vals, mean_val)
            stats = {
                'type': 'numeric',
                'mean': mean_val,
                'std': std_val,
                'min': min(numeric_vals),
                'max': max(numeric_vals)
            }
        else:
            stats = {'type': 'numeric', 'mean': 0, 'std': 1, 'min': 0, 'max': 1}
    else:  # Categorical or string
        stats = {
            'type': 'categorical',
            'unique_values': list(unique_values),
            'value_counts': defaultdict(int)
        }
        for val in values:
            if val.strip():
                stats['value_counts'][val] += 1
                
    return stats

def parse_csv_data(file_path=None, csv_string=None):
    """Parse CSV data from either a file or string with type detection"""
    try:
        if file_path:
            f = open(file_path, 'r', newline='', encoding='utf-8')
        else:
            f = io.StringIO(csv_string)
        
        # Try different delimiters
        sample = f.read(1024)
        f.seek(0)
        
        try:
            dialect = csv.Sniffer().sniff(sample)
            has_header = csv.Sniffer().has_header(sample)
        except:
            dialect = 'excel'  # fallback to excel dialect
            has_header = True
        
        reader = csv.reader(f, dialect)
        
        if has_header:
            header = next(reader)
        else:
            # If no header, create generic column names
            first_row = next(reader)
            header = [f'Column_{i+1}' for i in range(len(first_row))]
            # Rewind to include first row in data
            f.seek(0)
            reader = csv.reader(f, dialect)
            next(reader)  # skip header we just created
        
        # Read all data
        data = []
        for row in reader:
            if not row:  # Skip empty rows
                continue
            data.append(row)
            
        if not data:
            raise ValueError("No data rows found")
        
        # Transpose to get columns
        columns = list(zip(*data))
        column_stats = []
        
        # Analyze each column
        for col in columns:
            column_stats.append(analyze_column(col))
            
        return header, data, column_stats
    except Exception as e:
        raise ValueError(f"Error parsing CSV data: {str(e)}")
    finally:
        if file_path and 'f' in locals():
            f.close()

def generate_synthetic_value(col_stats):
    """Generate a synthetic value based on column statistics"""
    if col_stats['type'] == 'numeric':
        if col_stats['std'] == 0:
            return str(round(col_stats['mean'], 4))
        val = random.gauss(col_stats['mean'], col_stats['std'])
        # Clip to min/max range
        val = max(col_stats['min'], min(col_stats['max'], val))
        return str(round(val, 4))
    else:  # categorical
        values = col_stats['unique_values']
        if not values:
            return ''
        # Weighted random choice based on frequency
        return random.choices(
            list(col_stats['value_counts'].keys()),
            weights=list(col_stats['value_counts'].values()),
            k=1
        )[0]

def generate_synthetic_data(header, data, column_stats, num_rows=None):
    """Generate synthetic data preserving original patterns"""
    if num_rows is None:
        num_rows = len(data)
    
    synthetic_data = []
    for _ in range(num_rows):
        row = []
        for col_idx in range(len(header)):
            row.append(generate_synthetic_value(column_stats[col_idx]))
        synthetic_data.append(row)
    
    return synthetic_data

def write_csv_string(header, data):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(header)
    writer.writerows(data)
    return output.getvalue()

class SimpleHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        """Override to suppress standard request logging"""
        pass

    def do_GET(self):
        try:
            if self.path.startswith('/download/'):
                filename = self.path.split('/')[-1]
                if not filename or filename == 'download':
                    raise ValueError("Invalid filename")
                
                file_path = os.path.join(UPLOAD_DIR, filename)
                if not os.path.exists(file_path):
                    raise FileNotFoundError("File not found")
                
                with open(file_path, 'rb') as f:
                    content = f.read()
                
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.end_headers()
                self.wfile.write(content)
                return

            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            
            html_content = """
            <html>
                <head>
                    <title>Synthetic Data Generator</title>
                    <style>
                        body { 
                            font-family: Arial, sans-serif; 
                            max-width: 800px; 
                            margin: 0 auto; 
                            padding: 20px;
                            background: linear-gradient(135deg, #e0f7fa, #b2ebf2, #80deea);
                            min-height: 100vh;
                        }
                        .container {
                            background-color: white;
                            padding: 30px;
                            border-radius: 10px;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        }
                        h2 { color: #00796b; }
                        form { margin: 20px 0; }
                        .success { color: #2e7d32; }
                        .error { color: #c62828; }
                        .instructions { 
                            background: #f5f5f5; 
                            padding: 15px; 
                            border-radius: 5px; 
                            margin-bottom: 20px;
                            border-left: 4px solid #00796b;
                        }
                        input[type="file"], textarea {
                            padding: 10px;
                            border: 1px solid #b2dfdb;
                            border-radius: 4px;
                            background: #f5f5f5;
                            width: 100%;
                            box-sizing: border-box;
                        }
                        button {
                            background-color: #00796b;
                            color: white;
                            padding: 10px 20px;
                            border: none;
                            border-radius: 4px;
                            cursor: pointer;
                            font-size: 16px;
                            margin-top: 10px;
                        }
                        button:hover {
                            background-color: #00695c;
                        }
                        a {
                            color: #00796b;
                            text-decoration: none;
                            font-weight: bold;
                        }
                        a:hover {
                            text-decoration: underline;
                        }
                        ul {
                            padding-left: 20px;
                        }
                        li {
                            margin-bottom: 8px;
                        }
                        .tab-content {
                            display: none;
                        }
                        .tab-content.active {
                            display: block;
                        }
                        .tab-links {
                            display: flex;
                            margin-bottom: 20px;
                        }
                        .tab-link {
                            padding: 10px 20px;
                            background: #b2dfdb;
                            margin-right: 5px;
                            cursor: pointer;
                            border-radius: 4px 4px 0 0;
                        }
                        .tab-link.active {
                            background: #00796b;
                            color: white;
                        }
                    </style>
                    <script>
                        function openTab(evt, tabName) {
                            var i, tabcontent, tablinks;
                            tabcontent = document.getElementsByClassName("tab-content");
                            for (i = 0; i < tabcontent.length; i++) {
                                tabcontent[i].classList.remove("active");
                            }
                            tablinks = document.getElementsByClassName("tab-link");
                            for (i = 0; i < tablinks.length; i++) {
                                tablinks[i].classList.remove("active");
                            }
                            document.getElementById(tabName).classList.add("active");
                            evt.currentTarget.classList.add("active");
                        }
                    </script>
                </head>
                <body>
                    <div class="container">
                        <h2>📄 Advanced Synthetic Data Generator</h2>
                        
                        <div class="tab-links">
                            <div class="tab-link active" onclick="openTab(event, 'upload-tab')">Upload File</div>
                            <div class="tab-link" onclick="openTab(event, 'paste-tab')">Paste Data</div>
                        </div>
                        
                        <div id="upload-tab" class="tab-content active">
                            <div class="instructions">
                                <h3>Instructions:</h3>
                                <ul>
                                    <li>Upload a CSV file containing your data</li>
                                    <li>Can contain both numeric and text columns</li>
                                    <li>First row should be column headers</li>
                                    <li>Empty cells will be handled automatically</li>
                                    <li>Maximum file size: 10MB</li>
                                </ul>
                            </div>
                            <form enctype="multipart/form-data" method="POST">
                                <input type="file" name="file" accept=".csv" required>
                                <button type="submit">Generate Synthetic Data</button>
                            </form>
                        </div>
                        
                        <div id="paste-tab" class="tab-content">
                            <div class="instructions">
                                <h3>Instructions:</h3>
                                <ul>
                                    <li>Paste your CSV data below</li>
                                    <li>Can contain both numbers and text</li>
                                    <li>First row should be column headers</li>
                                    <li>Separate values with commas</li>
                                </ul>
                            </div>
                            <form action="/generate" method="post">
                                <textarea name="csv_data" rows="10" required placeholder="Name,Age,City\nAlice,25,New York\nBob,30,Chicago"></textarea>
                                <div style="margin-top: 10px;">
                                    <label for="num_rows">Number of synthetic rows:</label>
                                    <input type="number" id="num_rows" name="num_rows" min="1" value="100" style="width: 80px;">
                                </div>
                                <button type="submit">Generate Synthetic Data</button>
                            </form>
                        </div>
                    </div>
                </body>
            </html>
            """
            self.wfile.write(html_content.encode('utf-8'))
            
        except Exception as e:
            self.send_error(400, str(e))

    def do_POST(self):
        parsed_path = urlparse(self.path)
        
        # Handle direct CSV data processing
        if parsed_path.path == '/generate':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length == 0:
                    raise ValueError("No data received")
                
                # Get the form data
                post_data = self.rfile.read(content_length).decode('utf-8')
                post_vars = parse_qs(post_data)
                csv_data = post_vars.get('csv_data', [''])[0]
                num_rows = int(post_vars.get('num_rows', ['100'])[0])
                
                if not csv_data.strip():
                    raise ValueError("No CSV data provided")
                
                # Process the CSV data
                header, data, column_stats = parse_csv_data(csv_string=csv_data)
                synthetic_data = generate_synthetic_data(header, data, column_stats, num_rows)
                synthetic_csv = write_csv_string(header, synthetic_data)
                
                # Create download filename
                synthetic_filename = f"synthetic_data_{random.randint(1000,9999)}.csv"
                synthetic_path = os.path.join(UPLOAD_DIR, synthetic_filename)
                with open(synthetic_path, 'w', newline='', encoding='utf-8') as f:
                    f.write(synthetic_csv)

                html = f"""
                <html>
                    <head>
                        <title>Synthetic Data Generated</title>
                        <style>
                            body {{ 
                                font-family: Arial, sans-serif; 
                                max-width: 800px; 
                                margin: 0 auto; 
                                padding: 20px;
                                background: linear-gradient(135deg, #e0f7fa, #b2ebf2, #80deea);
                                min-height: 100vh;
                            }}
                            .container {{
                                background-color: white;
                                padding: 30px;
                                border-radius: 10px;
                                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                            }}
                            .success {{ color: #2e7d32; }}
                            a {{
                                color: #00796b;
                                text-decoration: none;
                                font-weight: bold;
                            }}
                            a:hover {{
                                text-decoration: underline;
                            }}
                            p {{
                                margin: 10px 0;
                            }}
                            table {{
                                width: 100%;
                                border-collapse: collapse;
                                margin: 20px 0;
                            }}
                            th, td {{
                                border: 1px solid #ddd;
                                padding: 8px;
                                text-align: left;
                            }}
                            th {{
                                background-color: #f2f2f2;
                            }}
                            tr:nth-child(even) {{
                                background-color: #f9f9f9;
                            }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h2 class="success">✅ Synthetic Data Generated</h2>
                            <p>Rows generated: {len(synthetic_data)}</p>
                            <p>Columns processed: {len(header)}</p>
                            
                            <div style="max-height: 300px; overflow-y: auto;">
                                <table>
                                    <thead>
                                        <tr>
                                            {''.join(f'<th>{h}</th>' for h in header)}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {''.join(f'<tr>{''.join(f'<td>{val}</td>' for val in row)}</tr>' for row in synthetic_data[:5])}
                                        <tr><td colspan="{len(header)}" style="text-align: center;">... and {len(synthetic_data)-5} more rows</td></tr>
                                    </tbody>
                                </table>
                            </div>
                            
                            <a href="/download/{synthetic_filename}" download>⬇️ Download Full Dataset</a><br/><br/>
                            <a href="/">🔙 Generate More Data</a>
                        </div>
                    </body>
                </html>
                """
                
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html.encode('utf-8'))
                return
                
            except Exception as e:
                error_trace = traceback.format_exc()
                print(f"Error processing request:\n{error_trace}")
                
                html = f"""
                <html>
                    <head>
                        <title>Error</title>
                        <style>
                            body {{ 
                                font-family: Arial, sans-serif; 
                                max-width: 800px; 
                                margin: 0 auto; 
                                padding: 20px;
                                background: linear-gradient(135deg, #e0f7fa, #b2ebf2, #80deea);
                                min-height: 100vh;
                            }}
                            .container {{
                                background-color: white;
                                padding: 30px;
                                border-radius: 10px;
                                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                            }}
                            .error {{ color: #c62828; }}
                            a {{
                                color: #00796b;
                                text-decoration: none;
                                font-weight: bold;
                            }}
                            a:hover {{
                                text-decoration: underline;
                            }}
                            ul {{
                                padding-left: 20px;
                            }}
                            li {{
                                margin-bottom: 8px;
                            }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h2 class="error">⚠️ Error Processing Request</h2>
                            <p><strong>Error:</strong> {str(e)}</p>
                            <p>Please check:</p>
                            <ul>
                                <li>Your data is properly formatted as CSV</li>
                                <li>The first row contains headers</li>
                                <li>Values are properly separated</li>
                            </ul>
                            <a href="/">🔙 Try Again</a>
                        </div>
                    </body>
                </html>
                """
                
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html.encode('utf-8'))
                return

        # Handle file upload processing with improved parsing
        try:
            content_type = self.headers.get('Content-Type', '')
            content_length = int(self.headers.get('Content-Length', 0))
            
            if not content_type or 'multipart/form-data' not in content_type:
                raise ValueError("Invalid content type. Must be multipart/form-data")
                
            if content_length == 0:
                raise ValueError("No content received. Please select a file to upload.")

            # Read the entire body
            post_data = self.rfile.read(content_length)
            
            # Create proper email message with headers for parsing
            headers = f"Content-Type: {content_type}\r\nContent-Length: {content_length}\r\n\r\n"
            msg = BytesParser(policy=default).parsebytes(
                headers.encode('utf-8') + post_data
            )

            # Debug: Print parts of the message
            print("\nReceived form parts:")
            for part in msg.iter_parts():
                print(f"Part: {part.get_content_type()}, {part.get_content_disposition()}")
                if part.get_content_disposition() == 'form-data':
                    print(f"Name: {part.get_param('name', header='content-disposition')}")
                    print(f"Filename: {part.get_filename()}")

            # Find the file part
            file_part = None
            for part in msg.iter_parts():
                if (part.get_content_disposition() == 'form-data' and 
                    part.get_param('name', header='content-disposition') == 'file' and 
                    part.get_filename()):
                    file_part = part
                    break

            if not file_part:
                raise ValueError("No file was uploaded or the form field name is incorrect")

            filename = sanitize_filename(file_part.get_filename())
            if not filename:
                raise ValueError("Invalid filename")

            # Ensure filename ends with .csv
            if not filename.lower().endswith('.csv'):
                filename += '.csv'

            file_data = file_part.get_payload(decode=True)
            file_path = os.path.join(UPLOAD_DIR, filename)

            # Debug file info
            print(f"File upload debug - Filename: {filename}")
            print(f"File size: {len(file_data)} bytes")
            if len(file_data) < 1000:  # Only print small files completely
                print(f"File content (first 1000 chars): {file_data[:1000].decode('utf-8', errors='replace')}")

            # Save the uploaded file
            with open(file_path, 'wb') as f:
                f.write(file_data)

            # Process the CSV file
            header, data, column_stats = parse_csv_data(file_path=file_path)
            synthetic_data = generate_synthetic_data(header, data, column_stats)
            synthetic_csv = write_csv_string(header, synthetic_data)

            synthetic_filename = f"synthetic_{filename}"
            synthetic_path = os.path.join(UPLOAD_DIR, synthetic_filename)
            with open(synthetic_path, 'w', newline='', encoding='utf-8') as f:
                f.write(synthetic_csv)

            html = f"""
            <html>
                <head>
                    <title>Synthetic Data Generated</title>
                    <style>
                        body {{ 
                            font-family: Arial, sans-serif; 
                            max-width: 800px; 
                            margin: 0 auto; 
                            padding: 20px;
                            background: linear-gradient(135deg, #e0f7fa, #b2ebf2, #80deea);
                            min-height: 100vh;
                        }}
                        .container {{
                            background-color: white;
                            padding: 30px;
                            border-radius: 10px;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        }}
                        .success {{ color: #2e7d32; }}
                        a {{
                            color: #00796b;
                            text-decoration: none;
                            font-weight: bold;
                        }}
                        a:hover {{
                            text-decoration: underline;
                        }}
                        p {{
                            margin: 10px 0;
                        }}
                        table {{
                            width: 100%;
                            border-collapse: collapse;
                            margin: 20px 0;
                        }}
                        th, td {{
                            border: 1px solid #ddd;
                            padding: 8px;
                            text-align: left;
                        }}
                        th {{
                            background-color: #f2f2f2;
                        }}
                        tr:nth-child(even) {{
                            background-color: #f9f9f9;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h2 class="success">✅ Synthetic Data Generated</h2>
                        <p>Original file: {filename}</p>
                        <p>Rows generated: {len(synthetic_data)}</p>
                        <p>Columns processed: {len(header)}</p>
                        
                        <div style="max-height: 300px; overflow-y: auto;">
                            <table>
                                <thead>
                                    <tr>
                                        {''.join(f'<th>{h}</th>' for h in header)}
                                    </tr>
                                </thead>
                                <tbody>
                                    {''.join(f'<tr>{''.join(f'<td>{val}</td>' for val in row)}</tr>' for row in synthetic_data[:5])}
                                    <tr><td colspan="{len(header)}" style="text-align: center;">... and {len(synthetic_data)-5} more rows</td></tr>
                                </tbody>
                            </table>
                        </div>
                        
                        <a href="/download/{synthetic_filename}" download>⬇️ Download Full Dataset</a><br/><br/>
                        <a href="/">🔙 Upload Another File</a>
                    </div>
                </body>
            </html>
            """
            
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))

        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Error processing request:\n{error_trace}")
            
            html = f"""
            <html>
                <head>
                    <title>Error</title>
                    <style>
                        body {{ 
                            font-family: Arial, sans-serif; 
                            max-width: 800px; 
                            margin: 0 auto; 
                            padding: 20px;
                            background: linear-gradient(135deg, #e0f7fa, #b2ebf2, #80deea);
                            min-height: 100vh;
                        }}
                        .container {{
                            background-color: white;
                            padding: 30px;
                            border-radius: 10px;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                        }}
                        .error {{ color: #c62828; }}
                        a {{
                            color: #00796b;
                            text-decoration: none;
                            font-weight: bold;
                        }}
                        a:hover {{
                            text-decoration: underline;
                        }}
                        ul {{
                            padding-left: 20px;
                        }}
                        li {{
                            margin-bottom: 8px;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h2 class="error">⚠️ Error Processing Request</h2>
                        <p><strong>Error:</strong> {str(e)}</p>
                        <p>Please check:</p>
                        <ul>
                            <li>You selected a file before submitting</li>
                            <li>The file is a valid CSV file</li>
                            <li>The file size is not too large (try under 10MB)</li>
                            <li>The upload directory has write permissions</li>
                        </ul>
                        <a href="/">🔙 Try Again</a>
                    </div>
                </body>
            </html>
            """
            
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))

if __name__ == '__main__':
    PORT = 8080
    print(f"🌐 Server running at http://localhost:{PORT}")
    print(f"📁 Upload directory: {os.path.abspath(UPLOAD_DIR)}")
    print("Press Ctrl+C to stop the server")
    
    try:
        server = HTTPServer(('', PORT), SimpleHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down...")
    except Exception as e:
        print(f"Server error: {str(e)}")