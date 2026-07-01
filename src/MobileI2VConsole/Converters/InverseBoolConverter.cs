using System.Globalization;

namespace MobileI2VConsole.Converters;

/// <summary>
/// Inverts a boolean value. Used for XAML bindings where
/// we need the opposite of a bound bool (e.g., show when not loading).
/// </summary>
public class InverseBoolConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is bool boolValue)
            return !boolValue;
        return value;
    }

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
        if (value is bool boolValue)
            return !boolValue;
        return value;
    }
}
