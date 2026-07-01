using System.Globalization;

namespace MobileI2VConsole.Converters;

/// <summary>
/// Returns true when the value is not null. Used to show/hide
/// elements based on whether a property is set.
/// </summary>
public class NotNullToBoolConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        return value != null;
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        throw new NotImplementedException();
    }
}
