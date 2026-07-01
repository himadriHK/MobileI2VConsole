using MobileI2VConsole.ViewModels;

namespace MobileI2VConsole.Views;

public partial class ResultPage : ContentPage
{
    public ResultPage(ResultViewModel viewModel)
    {
        InitializeComponent();
        BindingContext = viewModel;
    }
}
