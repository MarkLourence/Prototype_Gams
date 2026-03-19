from app_controller import AppController  # Import the AppController class, which manages the overall application flow and screens


if __name__ == "__main__":  # Check if this script is being run directly (not imported as a module)
    app = AppController()  # Create an instance of the AppController to handle the application
    app.run()  # Start the main application loop, which will manage screens and user interactions
