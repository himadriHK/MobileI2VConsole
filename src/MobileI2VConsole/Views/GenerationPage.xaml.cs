using MobileI2VConsole.ViewModels;

namespace MobileI2VConsole.Views;

public partial class GenerationPage : ContentPage
{
    public GenerationPage(GenerationViewModel viewModel)
    {
        InitializeComponent();
        BindingContext = viewModel;
    }
}
