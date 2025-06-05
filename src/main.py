import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
from pipeline import upload_image, enhance_img, preform_ocr, export_ready_data, export_pdf, export_word
import utils


class OCRDocumentProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Document Processor")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.current_image = None
        self.processed_image = None
        self.ocr_results = None
        self.export_data = None
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = tk.Label(main_frame, text="OCR Document Processor", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel for controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Image processing buttons
        self.upload_btn = ttk.Button(control_frame, text="📁 Upload & Process Image", 
                                    command=self.upload_and_process_image, width=25)
        self.upload_btn.grid(row=0, column=0, pady=5, sticky=tk.W+tk.E)
        
        self.enhance_btn = ttk.Button(control_frame, text="✨ Enhance Image", 
                                     command=self.enhance_image, width=25, state=tk.DISABLED)
        self.enhance_btn.grid(row=1, column=0, pady=5, sticky=tk.W+tk.E)
        
        self.ocr_btn = ttk.Button(control_frame, text="🔍 Perform OCR", 
                                 command=self.perform_ocr, width=25, state=tk.DISABLED)
        self.ocr_btn.grid(row=2, column=0, pady=5, sticky=tk.W+tk.E)
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(row=3, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Export buttons
        export_label = tk.Label(control_frame, text="Export Options:", font=('Arial', 10, 'bold'))
        export_label.grid(row=4, column=0, pady=(5, 10), sticky=tk.W)
        
        self.export_pdf_btn = ttk.Button(control_frame, text="📄 Export to PDF", 
                                        command=self.export_to_pdf, width=25, state=tk.DISABLED)
        self.export_pdf_btn.grid(row=5, column=0, pady=2, sticky=tk.W+tk.E)
        
        self.export_word_btn = ttk.Button(control_frame, text="📝 Export to Word", 
                                         command=self.export_to_word, width=25, state=tk.DISABLED)
        self.export_word_btn.grid(row=6, column=0, pady=2, sticky=tk.W+tk.E)
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.grid(row=7, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Status label
        self.status_label = tk.Label(control_frame, text="Ready", 
                                    fg='green', font=('Arial', 9))
        self.status_label.grid(row=8, column=0, pady=5, sticky=tk.W)
        
        # Middle panel for image display
        image_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        image_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        
        # Image display with scrollbars
        self.image_canvas = tk.Canvas(image_frame, bg='white', width=400, height=500)
        self.image_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars for image canvas
        v_scrollbar = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.image_canvas.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=self.image_canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.image_canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # Right panel for OCR results
        results_frame = ttk.LabelFrame(main_frame, text="OCR Results", padding="10")
        results_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Text area for OCR results
        self.results_text = scrolledtext.ScrolledText(results_frame, width=40, height=30, 
                                                     font=('Courier', 10))
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Clear results button
        self.clear_btn = ttk.Button(results_frame, text="Clear Results", 
                                   command=self.clear_results)
        self.clear_btn.grid(row=1, column=0, pady=(5, 0), sticky=tk.E)
        
    def update_status(self, message, color='black'):
        """Update status label with message and color"""
        self.status_label.config(text=message, fg=color)
        self.root.update_idletasks()
        
    def start_progress(self):
        """Start progress bar animation"""
        self.progress.start(10)
        
    def stop_progress(self):
        """Stop progress bar animation"""
        self.progress.stop()
        
    def display_image(self, image):
        """Display image on canvas"""
        try:
            # Convert OpenCV image to PIL Image
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
            else:
                pil_image = Image.fromarray(image)
            
            # Resize image to fit canvas while maintaining aspect ratio
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:  # Canvas is initialized
                img_width, img_height = pil_image.size
                scale = min(canvas_width/img_width, canvas_height/img_height, 1.0)
                
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage and display
            self.photo = ImageTk.PhotoImage(pil_image)
            
            # Clear canvas and add image
            self.image_canvas.delete("all")
            self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            
            # Update scroll region
            self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all"))
            
        except Exception as e:
            messagebox.showerror("Display Error", f"Error displaying image: {str(e)}")
    
    def upload_and_process_image(self):
        """Upload and process image using pipeline"""
        def process():
            try:
                self.start_progress()
                self.update_status("Processing image...", 'blue')
                
                # Use pipeline function to upload and process image
                processed_img = upload_image()
                
                if processed_img is not None:
                    self.current_image = processed_img
                    self.processed_image = processed_img.copy()
                    
                    # Display processed image
                    self.root.after(0, lambda: self.display_image(processed_img))
                    
                    # Enable next step buttons
                    self.root.after(0, lambda: self.enhance_btn.config(state=tk.NORMAL))
                    self.root.after(0, lambda: self.ocr_btn.config(state=tk.NORMAL))
                    
                    self.root.after(0, lambda: self.update_status("Image processed successfully!", 'green'))
                else:
                    self.root.after(0, lambda: self.update_status("Failed to process image", 'red'))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Processing Error", 
                                                               f"Error processing image: {str(e)}"))
                self.root.after(0, lambda: self.update_status("Processing failed", 'red'))
            finally:
                self.root.after(0, self.stop_progress)
        
        # Run in separate thread to prevent GUI freezing
        threading.Thread(target=process, daemon=True).start()
    
    def enhance_image(self):
        """Enhance current image"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
            
        def enhance():
            try:
                self.start_progress()
                self.update_status("Enhancing image...", 'blue')
                
                # Use pipeline function to enhance image
                enhanced_img = enhance_img(self.current_image)
                
                if enhanced_img is not None:
                    self.processed_image = enhanced_img
                    
                    # Display enhanced image
                    self.root.after(0, lambda: self.display_image(enhanced_img))
                    self.root.after(0, lambda: self.update_status("Image enhanced successfully!", 'green'))
                else:
                    self.root.after(0, lambda: self.update_status("Failed to enhance image", 'red'))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Enhancement Error", 
                                                               f"Error enhancing image: {str(e)}"))
                self.root.after(0, lambda: self.update_status("Enhancement failed", 'red'))
            finally:
                self.root.after(0, self.stop_progress)
        
        threading.Thread(target=enhance, daemon=True).start()
    
    def perform_ocr(self):
        """Perform OCR on current image"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
            
        def ocr():
            try:
                self.start_progress()
                self.update_status("Performing OCR...", 'blue')
                
                # Use pipeline function to perform OCR
                results = preform_ocr(self.current_image)
                
                if results is not None:
                    self.ocr_results = results
                    
                    # Prepare export data
                    height, width = self.current_image.shape[:2]
                    if len(self.current_image.shape) == 3:
                        image_shape = (height, width)
                    else:
                        image_shape = (height, width)
                    
                    self.export_data = export_ready_data(results, self.current_image)
                    
                    # Display OCR results
                    self.root.after(0, self.display_ocr_results)
                    
                    # Enable export buttons
                    self.root.after(0, lambda: self.export_pdf_btn.config(state=tk.NORMAL))
                    self.root.after(0, lambda: self.export_word_btn.config(state=tk.NORMAL))
                    
                    self.root.after(0, lambda: self.update_status("OCR completed successfully!", 'green'))
                else:
                    self.root.after(0, lambda: self.update_status("OCR failed", 'red'))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("OCR Error", 
                                                               f"Error performing OCR: {str(e)}"))
                self.root.after(0, lambda: self.update_status("OCR failed", 'red'))
            finally:
                self.root.after(0, self.stop_progress)
        
        threading.Thread(target=ocr, daemon=True).start()
    
    def display_ocr_results(self):
        """Display OCR results in text area"""
        if self.ocr_results is None:
            return
            
        self.results_text.delete(1.0, tk.END)
        
        # Add header
        self.results_text.insert(tk.END, "=== OCR RESULTS ===\n\n")
        
        # Display each text item with confidence
        for i, result in enumerate(self.ocr_results, 1):
            text = result.get('text', '')
            confidence = result.get('confidence', 0)
            bbox = result.get('bbox', (0, 0, 0, 0))
            
            self.results_text.insert(tk.END, f"{i}. Text: '{text}'\n")
            self.results_text.insert(tk.END, f"   Confidence: {confidence:.1f}%\n")
            self.results_text.insert(tk.END, f"   Position: {bbox}\n\n")
        
        # Add summary
        total_items = len(self.ocr_results)
        avg_confidence = sum(result.get('confidence', 0) for result in self.ocr_results) / total_items if total_items > 0 else 0
        
        self.results_text.insert(tk.END, f"\n=== SUMMARY ===\n")
        self.results_text.insert(tk.END, f"Total text items found: {total_items}\n")
        self.results_text.insert(tk.END, f"Average confidence: {avg_confidence:.1f}%\n")
    
    def export_to_pdf(self):
        """Export OCR results to PDF"""
        if self.export_data is None or self.current_image is None:
            messagebox.showwarning("No Data", "Please perform OCR first.")
            return
            
        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Save PDF as..."
        )
        
        if not file_path:
            return
            
        def export():
            try:
                self.start_progress()
                self.update_status("Exporting to PDF...", 'blue')
                
                # Use pipeline function to export PDF
                export_pdf(self.export_data, self.current_image, file_path)
                
                self.root.after(0, lambda: self.update_status("PDF exported successfully!", 'green'))
                self.root.after(0, lambda: messagebox.showinfo("Export Complete", 
                                                              f"PDF saved to: {file_path}"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Export Error", 
                                                               f"Error exporting PDF: {str(e)}"))
                self.root.after(0, lambda: self.update_status("PDF export failed", 'red'))
            finally:
                self.root.after(0, self.stop_progress)
        
        threading.Thread(target=export, daemon=True).start()
    
    def export_to_word(self):
        """Export OCR results to Word document"""
        if self.export_data is None:
            messagebox.showwarning("No Data", "Please perform OCR first.")
            return
            
        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".docx",
            filetypes=[("Word documents", "*.docx")],
            title="Save Word document as..."
        )
        
        if not file_path:
            return
            
        def export():
            try:
                self.start_progress()
                self.update_status("Exporting to Word...", 'blue')
                
                # Use pipeline function to export Word
                export_word(self.export_data, file_path)
                
                self.root.after(0, lambda: self.update_status("Word document exported successfully!", 'green'))
                self.root.after(0, lambda: messagebox.showinfo("Export Complete", 
                                                              f"Word document saved to: {file_path}"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Export Error", 
                                                               f"Error exporting Word: {str(e)}"))
                self.root.after(0, lambda: self.update_status("Word export failed", 'red'))
            finally:
                self.root.after(0, self.stop_progress)
        
        threading.Thread(target=export, daemon=True).start()
    
    def clear_results(self):
        """Clear OCR results"""
        self.results_text.delete(1.0, tk.END)
        self.ocr_results = None
        self.export_data = None
        self.export_pdf_btn.config(state=tk.DISABLED)
        self.export_word_btn.config(state=tk.DISABLED)
        self.update_status("Results cleared", 'blue')


def main():
    """Main function to run the GUI application"""
    try:
        # Create main window
        root = tk.Tk()
        
        # Create application instance
        app = OCRDocumentProcessorGUI(root)
        
        # Start GUI event loop
        root.mainloop()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application: {str(e)}")


if __name__ == "__main__":
    main()