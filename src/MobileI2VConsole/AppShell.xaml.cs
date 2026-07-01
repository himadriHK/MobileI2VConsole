using MobileI2VConsole.Views;

namespace MobileI2VConsole;

public partial class AppShell : Shell
{
    public AppShell()
    {
        InitializeComponent();
        Routing.RegisterRoute("generation", typeof(GenerationPage));
        Routing.RegisterRoute("result", typeof(ResultPage));
    }
}
